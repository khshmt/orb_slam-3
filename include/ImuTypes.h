/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * ImuTypes.h — ORB-SLAM3
 *
 * This header defines all IMU-related data structures used throughout ORB-SLAM3:
 *   - Point       : a single raw IMU measurement (gyro + accel + timestamp)
 *   - Bias        : gyro and accelerometer bias estimates
 *   - Calib       : IMU calibration (extrinsics + noise covariance matrices)
 *   - IntegratedRotation : result of integrating one gyro measurement over dt
 *   - Preintegrated      : full IMU preintegration between two keyframes,
 *                          including delta rotation/velocity/position,
 *                          bias Jacobians, and covariance propagation
 *
 * The preintegration theory follows:
 *   Forster et al., "On-Manifold Preintegration for Real-Time
 *   Visual-Inertial Odometry", IEEE TRO 2017.
 */

#ifndef IMUTYPES_H
#define IMUTYPES_H

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <mutex>

#include "SerializationUtils.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace ORB_SLAM3
{

namespace IMU
{

// Standard gravity constant used throughout preintegration (m/s²)
const float GRAVITY_VALUE = 9.81;

// ---------------------------------------------------------------------------
// class Point
// ---------------------------------------------------------------------------
// Represents a single raw IMU measurement at one timestamp.
// Stores the 3-axis accelerometer reading (a), 3-axis gyroscope reading (w),
// and the timestamp (t) in seconds.
// ---------------------------------------------------------------------------
class Point
{
public:
    // Constructor from individual float components
    Point(const float &acc_x, const float &acc_y, const float &acc_z,
          const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
          const double &timestamp)
        : a(acc_x, acc_y, acc_z),
          w(ang_vel_x, ang_vel_y, ang_vel_z),
          t(timestamp) {}

    // Constructor from OpenCV Point3f (used when reading from ROS messages
    // or dataset loaders that provide cv::Point3f)
    Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp)
        : a(Acc.x, Acc.y, Acc.z),
          w(Gyro.x, Gyro.y, Gyro.z),
          t(timestamp) {}

public:
    Eigen::Vector3f a;   // Accelerometer reading [ax, ay, az] in m/s²  (body frame)
    Eigen::Vector3f w;   // Gyroscope reading     [wx, wy, wz] in rad/s (body frame)
    double t;            // Timestamp in seconds
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


// ---------------------------------------------------------------------------
// class Bias
// ---------------------------------------------------------------------------
// Holds the estimated bias for both gyroscope and accelerometer.
//
// IMU sensors have a slowly-varying offset (bias) on top of their true
// measurement. If not corrected, this bias integrates into position/velocity
// error over time.
//
// The bias is represented as 6 scalar values:
//   bax, bay, baz : accelerometer bias in m/s² per axis
//   bwx, bwy, bwz : gyroscope bias    in rad/s per axis
//
// During optimization (bundle adjustment), these bias values are refined.
// The preintegration stores the bias at the time of integration (b) and
// an updated bias (bu) so that bias corrections can be applied without
// re-integrating all the raw IMU measurements.
// ---------------------------------------------------------------------------
class Bias
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        // Serialize all 6 bias components for saving/loading map files
        ar & bax; ar & bay; ar & baz;
        ar & bwx; ar & bwy; ar & bwz;
    }

public:
    // Default constructor: zero bias (assumes perfect sensor initially)
    Bias() : bax(0), bay(0), baz(0), bwx(0), bwy(0), bwz(0) {}

    // Constructor with explicit bias values
    Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
         const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z)
        : bax(b_acc_x), bay(b_acc_y), baz(b_acc_z),
          bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z) {}

    void CopyFrom(Bias &b);

    friend std::ostream &operator<<(std::ostream &out, const Bias &b);

public:
    float bax, bay, baz;   // Accelerometer bias [m/s²]
    float bwx, bwy, bwz;   // Gyroscope bias     [rad/s]
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


// ---------------------------------------------------------------------------
// class Calib
// ---------------------------------------------------------------------------
// Stores the IMU calibration parameters:
//
//   mTbc : Transform from IMU body frame (b) to camera frame (c)
//          This is the extrinsic calibration — physically the rigid
//          transformation between where the IMU sits and where the camera sits.
//          Typically estimated with a tool like Kalibr.
//
//   mTcb : Inverse of mTbc (camera -> IMU body)
//
//   Cov  : 6x6 diagonal noise covariance matrix for the IMU measurements:
//          [ng², ng², ng², na², na², na²]
//           ^gyro noise density²    ^accel noise density²
//          These come from your Allan variance test results.
//
//   CovWalk : 6x6 diagonal covariance for bias random walk:
//             [ngw², ngw², ngw², naw², naw², naw²]
//              ^gyro random walk²    ^accel random walk²
//             Models how fast the bias drifts over time.
// ---------------------------------------------------------------------------
class Calib
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        serializeSophusSE3(ar, mTcb, version);
        serializeSophusSE3(ar, mTbc, version);
        ar & boost::serialization::make_array(Cov.diagonal().data(),     Cov.diagonal().size());
        ar & boost::serialization::make_array(CovWalk.diagonal().data(), CovWalk.diagonal().size());
        ar & mbIsSet;
    }

public:
    // Primary constructor: takes extrinsic transform and 4 noise scalars
    //   ng  = gyro noise density     (rad/s/sqrt(Hz))  — from Allan variance
    //   na  = accel noise density    (m/s²/sqrt(Hz))   — from Allan variance
    //   ngw = gyro random walk       (rad/s²/sqrt(Hz)) — from Allan variance
    //   naw = accel random walk      (m/s³/sqrt(Hz))   — from Allan variance
    Calib(const Sophus::SE3<float> &Tbc,
          const float &ng, const float &na,
          const float &ngw, const float &naw)
    {
        Set(Tbc, ng, na, ngw, naw);
    }

    Calib(const Calib &calib);
    Calib() { mbIsSet = false; }

    void Set(const Sophus::SE3<float> &sophTbc,
             const float &ng, const float &na,
             const float &ngw, const float &naw);

public:
    Sophus::SE3<float> mTcb;               // Transform: camera -> IMU body
    Sophus::SE3<float> mTbc;               // Transform: IMU body -> camera (extrinsics)
    Eigen::DiagonalMatrix<float, 6> Cov;      // Measurement noise covariance (diagonal)
    Eigen::DiagonalMatrix<float, 6> CovWalk;  // Bias random walk covariance  (diagonal)
    bool mbIsSet;                           // True once Set() has been called
};


// ---------------------------------------------------------------------------
// class IntegratedRotation
// ---------------------------------------------------------------------------
// Represents the result of integrating ONE gyroscope measurement over dt.
//
// This is the smallest unit of rotation integration. It computes:
//
//   deltaR : The rotation matrix R(t+dt) expressed relative to R(t),
//            computed from the Rodrigues / exponential map formula:
//
//            Let phi = (w - b_gyro) * dt   (bias-corrected angular velocity * dt)
//            Let theta = ||phi||            (rotation angle)
//            Let W = hat(phi)               (skew-symmetric matrix of phi)
//
//            If theta is very small (near zero rotation):
//                deltaR ≈ I + W            (first-order approximation)
//            Otherwise (Rodrigues formula):
//                deltaR = I + W*sin(θ)/θ + W²*(1-cos(θ))/θ²
//
//   rightJ : The right Jacobian of SO(3), Jr(phi).
//            This is needed for covariance propagation — it maps noise
//            in the rotation vector phi to noise in the rotation matrix deltaR.
//
//            If theta is very small:
//                Jr ≈ I
//            Otherwise:
//                Jr = I - W*(1-cos(θ))/θ² + W²*(θ-sin(θ))/θ³
// ---------------------------------------------------------------------------
class IntegratedRotation
{
public:
    IntegratedRotation() {}

    // angVel   : raw gyroscope measurement [wx, wy, wz] in rad/s
    // imuBias  : current gyroscope bias estimate to subtract from measurement
    // time     : integration timestep dt in seconds
    IntegratedRotation(const Eigen::Vector3f &angVel,
                       const Bias &imuBias,
                       const float &time);

public:
    float deltaT;           // Integration time dt (seconds)
    Eigen::Matrix3f deltaR; // Delta rotation matrix over this timestep (SO3)
    Eigen::Matrix3f rightJ; // Right Jacobian Jr(phi) at this step
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


// ---------------------------------------------------------------------------
// class Preintegrated
// ---------------------------------------------------------------------------
// The core IMU preintegration class. Accumulates all IMU measurements
// between two keyframes i and j into compact summary quantities:
//
//   dR  : Delta rotation     R_i^T * R_j          (3x3 rotation matrix)
//   dV  : Delta velocity     R_i^T * (v_j - v_i - g*dT)   (3-vector, m/s)
//   dP  : Delta position     R_i^T * (p_j - p_i - v_i*dT - 0.5*g*dT²) (3-vector, m)
//
// These are expressed in the body frame of keyframe i (NOT the world frame),
// which is the key advantage of preintegration: they are independent of the
// absolute pose and can be reused across multiple optimization iterations.
//
// Additionally stores:
//
//   Bias Jacobians (for first-order bias correction without re-integration):
//     JRg  : d(dR)/d(bias_gyro)   — how dR changes if gyro bias changes
//     JVg  : d(dV)/d(bias_gyro)
//     JVa  : d(dV)/d(bias_accel)
//     JPg  : d(dP)/d(bias_gyro)
//     JPa  : d(dP)/d(bias_accel)
//
//   Covariance (C, 15x15):
//     Blocks [0:9, 0:9]   — covariance of [dR, dV, dP] from measurement noise
//     Blocks [9:15, 9:15] — covariance of [bias_gyro, bias_accel] random walk
//
//   Info (15x15): Information matrix = C^{-1}, used in optimization residuals
//
//   b   : Original bias at integration time
//   bu  : Updated bias (from optimizer)
//   db  : db = bu - b (delta bias, used for first-order correction)
// ---------------------------------------------------------------------------
class Preintegrated
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & dT;
        ar & boost::serialization::make_array(C.data(),    C.size());
        ar & boost::serialization::make_array(Info.data(), Info.size());
        ar & boost::serialization::make_array(Nga.diagonal().data(),     Nga.diagonal().size());
        ar & boost::serialization::make_array(NgaWalk.diagonal().data(), NgaWalk.diagonal().size());
        ar & b;
        ar & boost::serialization::make_array(dR.data(),  dR.size());
        ar & boost::serialization::make_array(dV.data(),  dV.size());
        ar & boost::serialization::make_array(dP.data(),  dP.size());
        ar & boost::serialization::make_array(JRg.data(), JRg.size());
        ar & boost::serialization::make_array(JVg.data(), JVg.size());
        ar & boost::serialization::make_array(JVa.data(), JVa.size());
        ar & boost::serialization::make_array(JPg.data(), JPg.size());
        ar & boost::serialization::make_array(JPa.data(), JPa.size());
        ar & boost::serialization::make_array(avgA.data(), avgA.size());
        ar & boost::serialization::make_array(avgW.data(), avgW.size());
        ar & bu;
        ar & boost::serialization::make_array(db.data(), db.size());
        ar & mvMeasurements;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Main constructor: initializes with a known bias and calibration
    Preintegrated(const Bias &b_, const Calib &calib);

    // Copy constructor from pointer: duplicates all state from another
    // Preintegrated object (used when marginalizing keyframes)
    Preintegrated(Preintegrated *pImuPre);

    Preintegrated() {}
    ~Preintegrated() {}

    // Deep copy all fields from another Preintegrated object
    void CopyFrom(Preintegrated *pImuPre);

    // Reset all preintegrated quantities to zero / identity.
    // Called at the start of a new keyframe window.
    void Initialize(const Bias &b_);

    // Core function: integrate one new IMU measurement into the running totals.
    // Called once per IMU sample between keyframes.
    // acceleration : raw accel [ax, ay, az] m/s² in body frame
    // angVel       : raw gyro  [wx, wy, wz] rad/s in body frame
    // dt           : time since last measurement (seconds)
    void IntegrateNewMeasurement(const Eigen::Vector3f &acceleration,
                                  const Eigen::Vector3f &angVel,
                                  const float &dt);

    // Re-run integration from scratch using stored raw measurements
    // but with the updated bias bu. Called when optimizer changes the bias
    // significantly (beyond first-order correction range).
    void Reintegrate();

    // Merge a previous preintegration window into this one.
    // Used when two keyframes are merged (e.g. during loop closure).
    void MergePrevious(Preintegrated *pPrev);

    // Store a new updated bias from the optimizer.
    // Also computes db = bu - b for first-order bias correction.
    void SetNewBias(const Bias &bu_);

    // Return the bias difference between a given bias b_ and the
    // original integration bias b (used to check if re-integration is needed)
    IMU::Bias GetDeltaBias(const Bias &b_);

    // ---------------------------------------------------------------------------
    // Bias-corrected delta quantities (first-order linearization)
    // ---------------------------------------------------------------------------
    // These apply a first-order Taylor correction to dR/dV/dP for a new
    // bias b_ without full re-integration.  Valid when b_ is close to b.
    //
    // Corrected dR = dR * Exp(JRg * (b_.bwxyz - b.bwxyz))
    // Corrected dV = dV + JVg * dbg + JVa * dba
    // Corrected dP = dP + JPg * dbg + JPa * dba
    // ---------------------------------------------------------------------------
    Eigen::Matrix3f GetDeltaRotation(const Bias &b_);
    Eigen::Vector3f GetDeltaVelocity(const Bias &b_);
    Eigen::Vector3f GetDeltaPosition(const Bias &b_);

    // Same as above but using the internally stored db = bu - b
    Eigen::Matrix3f GetUpdatedDeltaRotation();
    Eigen::Vector3f GetUpdatedDeltaVelocity();
    Eigen::Vector3f GetUpdatedDeltaPosition();

    // Raw preintegrated values without any bias correction
    // (as computed at integration time with bias b)
    Eigen::Matrix3f GetOriginalDeltaRotation();
    Eigen::Vector3f GetOriginalDeltaVelocity();
    Eigen::Vector3f GetOriginalDeltaPosition();

    // Return bias vector db = [dbwx, dbwy, dbwz, dbax, dbay, dbaz] (6x1)
    Eigen::Matrix<float, 6, 1> GetDeltaBias();

    Bias GetOriginalBias();   // b  — bias used during integration
    Bias GetUpdatedBias();    // bu — latest bias from optimizer

    void printMeasurements() const {
        std::cout << "pint meas:\n";
        for (int i = 0; i < mvMeasurements.size(); i++)
            std::cout << "meas " << mvMeasurements[i].t << std::endl;
        std::cout << "end pint meas:\n";
    }

public:
    // -----------------------------------------------------------------------
    // Preintegrated state
    // -----------------------------------------------------------------------
    float dT;               // Total integrated time between keyframes i and j (seconds)

    // 15x15 covariance matrix, organized as:
    //   rows/cols [0:3]   -> delta rotation   dR
    //   rows/cols [3:6]   -> delta velocity   dV
    //   rows/cols [6:9]   -> delta position   dP
    //   rows/cols [9:12]  -> gyro  bias random walk
    //   rows/cols [12:15] -> accel bias random walk
    Eigen::Matrix<float, 15, 15> C;

    // Information matrix = C^{-1}. Used in optimization to weight residuals:
    //   cost = r^T * Info * r   (Mahalanobis distance)
    Eigen::Matrix<float, 15, 15> Info;

    // Measurement noise covariance (from Allan variance, diagonal 6x6):
    //   [ng², ng², ng², na², na², na²]
    Eigen::DiagonalMatrix<float, 6> Nga;

    // Bias random walk covariance (from Allan variance, diagonal 6x6):
    //   [ngw², ngw², ngw², naw², naw², naw²]
    Eigen::DiagonalMatrix<float, 6> NgaWalk;

    // -----------------------------------------------------------------------
    // Original bias at integration time
    // -----------------------------------------------------------------------
    Bias b;  // The bias b used when IntegrateNewMeasurement() was called

    // -----------------------------------------------------------------------
    // Preintegrated delta quantities (in body frame of keyframe i)
    // -----------------------------------------------------------------------
    Eigen::Matrix3f dR;        // Delta rotation:  R_i^T * R_j
    Eigen::Vector3f dV, dP;    // Delta velocity and position

    // -----------------------------------------------------------------------
    // Jacobians of delta quantities w.r.t. bias
    // Used for first-order bias correction (avoids full re-integration)
    // -----------------------------------------------------------------------
    Eigen::Matrix3f JRg;   // d(dR)/d(gyro_bias)   [3x3]
    Eigen::Matrix3f JVg;   // d(dV)/d(gyro_bias)   [3x3]
    Eigen::Matrix3f JVa;   // d(dV)/d(accel_bias)  [3x3]
    Eigen::Matrix3f JPg;   // d(dP)/d(gyro_bias)   [3x3]
    Eigen::Matrix3f JPa;   // d(dP)/d(accel_bias)  [3x3]

    // Running weighted averages of acceleration and angular velocity
    // (in world frame for accel, body frame for gyro)
    // Used for gravity initialization
    Eigen::Vector3f avgA, avgW;

private:
    // -----------------------------------------------------------------------
    // Updated bias (set by optimizer after each optimization round)
    // -----------------------------------------------------------------------
    Bias bu;  // Updated bias from most recent optimization iteration

    // Delta bias vector db = bu - b  [dbwx, dbwy, dbwz, dbax, dbay, dbaz]
    // Used for first-order correction without re-integration
    Eigen::Matrix<float, 6, 1> db;

    // -----------------------------------------------------------------------
    // Raw measurement storage
    // Used by Reintegrate() to redo integration with updated bias
    // -----------------------------------------------------------------------
    struct integrable
    {
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version)
        {
            ar & boost::serialization::make_array(a.data(), a.size());
            ar & boost::serialization::make_array(w.data(), w.size());
            ar & t;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        integrable() {}
        integrable(const Eigen::Vector3f &a_, const Eigen::Vector3f &w_, const float &t_)
            : a(a_), w(w_), t(t_) {}

        Eigen::Vector3f a;   // Raw accelerometer reading [m/s²]
        Eigen::Vector3f w;   // Raw gyroscope reading     [rad/s]
        float t;             // dt for this step          [seconds]
    };

    // All raw IMU measurements stored in order for potential re-integration
    std::vector<integrable> mvMeasurements;

    // Mutex to protect concurrent access (ORB-SLAM3 runs multiple threads)
    std::mutex mMutex;
};

// ---------------------------------------------------------------------------
// Lie Algebra utility functions (SO3)
// ---------------------------------------------------------------------------

// Right Jacobian of SO(3):
//   Jr(v) = I - (1-cos||v||)/||v||² * hat(v) + (||v||-sin||v||)/||v||³ * hat(v)²
// Maps perturbations in the Lie algebra to perturbations in the rotation matrix.
// Used in covariance propagation and bias Jacobian updates.
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v);

// Inverse Right Jacobian of SO(3):
//   Jr^{-1}(v) = I + hat(v)/2 + hat(v)²*(1/||v||² - (1+cos||v||)/(2*||v||*sin||v||))
// Used when converting from rotation matrix residuals back to Lie algebra.
Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v);

// Project a rotation matrix back onto SO(3) via SVD.
// Numerical integration causes R to drift away from being a valid rotation
// matrix (det != 1, not orthogonal). This function corrects that:
//   R_normalized = U * V^T   (from SVD: R = U * S * V^T)
Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R);

} // namespace IMU
} // namespace ORB_SLAM3

#endif // IMUTYPES_H
