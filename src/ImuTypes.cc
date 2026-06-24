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
 * ImuTypes.cc — ORB-SLAM3
 *
 * Implementation of all IMU preintegration mathematics.
 *
 * The core theory is on-manifold IMU preintegration (Forster et al. TRO 2017).
 * The key idea is to accumulate all IMU measurements between two keyframes
 * into compact delta quantities (dR, dV, dP) that are independent of the
 * absolute pose — so they don't need to be recomputed when poses change
 * during bundle adjustment, only when the bias estimate changes significantly.
 */

#include "ImuTypes.h"
#include "Converter.h"
#include "GeometricTools.h"
#include <iostream>

namespace ORB_SLAM3
{
namespace IMU
{

// Threshold below which a rotation angle is considered numerically zero.
// Avoids division by zero in Rodrigues formula and Jacobian computations.
const float eps = 1e-4;

// ---------------------------------------------------------------------------
// NormalizeRotation
// ---------------------------------------------------------------------------
// After many steps of numerical integration, the accumulated rotation matrix
// dR can drift away from being a valid member of SO(3) — it may lose
// orthogonality (R^T*R != I) or its determinant may deviate from 1.
//
// This function projects R back onto SO(3) using Singular Value Decomposition:
//   R = U * S * V^T  (SVD decomposition)
//   R_normalized = U * V^T
//
// This gives the closest valid rotation matrix to R in the Frobenius norm sense.
// ---------------------------------------------------------------------------
Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // U * V^T drops the singular values S (which should all be ~1 for a valid R)
    // and returns the nearest orthogonal matrix with det = +1
    return svd.matrixU() * svd.matrixV().transpose();
}

// ---------------------------------------------------------------------------
// RightJacobianSO3
// ---------------------------------------------------------------------------
// Right Jacobian is used to track how changing rotation angles (in the tangent space of SO(3)) affects the rotation matrix itself.
// 
// Computes the Right Jacobian of SO(3) at the rotation vector v = [x, y, z].
// The Right Jacobian Jr(v) relates a small perturbation delta_v in the
// rotation vector to the corresponding perturbation in the rotation matrix:
//
//   Exp(v + delta_v) ≈ Exp(v) * Exp(Jr(v) * delta_v)
//
// This is used in covariance propagation to map gyro noise (in rotation vector
// space) into noise in the rotation matrix (in SO(3) space).
//
// Formula (closed form):
//   W  = hat(v)              — skew-symmetric matrix of v
//   θ  = ||v||               — rotation angle
//
//   If θ < eps (near zero rotation):
//       Jr ≈ I               — identity (first-order approximation)
//   Else:
//       Jr = I - (1-cos θ)/θ² * W + (θ - sin θ)/θ³ * W²
// ---------------------------------------------------------------------------
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z)
{
    Eigen::Matrix3f I;
    I.setIdentity();

    const float d2 = x*x + y*y + z*z;   // θ² = ||v||²
    const float d  = sqrt(d2);           // θ  = ||v||

    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);  // skew-symmetric [v]×

    if (d < eps)
    {
        // Near-zero rotation: Jr ≈ I (higher order terms vanish)
        return I;
    }
    else
    {
        // Full closed-form right Jacobian
        return I - W * (1.0f - cos(d)) / d2
                 + W * W * (d - sin(d)) / (d2 * d);
    }
}

// Overload accepting a vector directly
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v)
{
    return RightJacobianSO3(v(0), v(1), v(2));
}

// ---------------------------------------------------------------------------
// InverseRightJacobianSO3
// ---------------------------------------------------------------------------
// Computes the inverse of the Right Jacobian: Jr^{-1}(v).
//
// Used when converting a rotation matrix residual back into a rotation
// vector perturbation (e.g. in the optimization cost function):
//
//   delta_v = Jr^{-1}(v) * Log(Exp(v)^T * Exp(v + delta_v))
//
// Formula:
//   If θ < eps:
//       Jr^{-1} ≈ I
//   Else:
//       Jr^{-1} = I + W/2 + W²*(1/θ² - (1+cos θ)/(2*θ*sin θ))
// ---------------------------------------------------------------------------
Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z)
{
    Eigen::Matrix3f I;
    I.setIdentity();

    const float d2 = x*x + y*y + z*z;
    const float d  = sqrt(d2);

    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);

    if (d < eps)
    {
        return I;
    }
    else
    {
        return I + W / 2.0f
                 + W * W * (1.0f / d2 - (1.0f + cos(d)) / (2.0f * d * sin(d)));
    }
}

Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v)
{
    return InverseRightJacobianSO3(v(0), v(1), v(2));
}

// ---------------------------------------------------------------------------
// IntegratedRotation constructor
// ---------------------------------------------------------------------------
// Integrates ONE gyroscope measurement over timestep dt to get:
//   - deltaR  : the rotation increment (a 3x3 rotation matrix in SO(3))
//   - rightJ  : the Right Jacobian at this step (for covariance propagation)
//
// Step 1: Subtract the gyro bias from the raw measurement and multiply by dt
//         to get the rotation vector phi = (w - b_gyro) * dt
//         phi represents "rotate by this angle around this axis"
//
// Step 2: Compute ||phi|| = theta (the rotation angle in radians)
//
// Step 3: Build the skew-symmetric matrix W = hat(phi)
//
// Step 4: Apply Rodrigues' rotation formula to get deltaR:
//   If theta < eps (negligible rotation):
//       deltaR ≈ I + W          (first-order: avoids numerical issues near zero)
//       rightJ = I
//   Else (general case):
//       deltaR = I + W*sin(θ)/θ + W²*(1-cos(θ))/θ²
//       rightJ = I - W*(1-cos(θ))/θ² + W²*(θ-sin(θ))/θ³
//
// Note: Rodrigues' formula is the closed-form expression for Exp(phi) on SO(3).
// It avoids the need to compute a full matrix exponential.
// ---------------------------------------------------------------------------
IntegratedRotation::IntegratedRotation(const Eigen::Vector3f &angVel,
                                        const Bias &imuBias,
                                        const float &time)
{
    // Bias-correct and scale by dt to get the rotation vector phi
    const float x = (angVel(0) - imuBias.bwx) * time;  // phi_x
    const float y = (angVel(1) - imuBias.bwy) * time;  // phi_y
    const float z = (angVel(2) - imuBias.bwz) * time;  // phi_z

    const float d2 = x*x + y*y + z*z;   // ||phi||²
    const float d  = sqrt(d2);           // ||phi|| = rotation angle theta

    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);  // skew-symmetric matrix [phi]×

    if (d < eps)
    {
        // Near-zero rotation: use first-order approximation
        // Exp(phi) ≈ I + [phi]×   for small ||phi||
        deltaR = Eigen::Matrix3f::Identity() + W;
        rightJ = Eigen::Matrix3f::Identity();  // Jr ≈ I for small phi
    }
    else
    {
        // Rodrigues' formula: Exp(phi) = I + W*sin(θ)/θ + W²*(1-cos(θ))/θ²
        deltaR = Eigen::Matrix3f::Identity()
                 + W * sin(d) / d
                 + W * W * (1.0f - cos(d)) / d2;

        // Right Jacobian: Jr(phi) = I - W*(1-cos θ)/θ² + W²*(θ-sin θ)/θ³
        rightJ = Eigen::Matrix3f::Identity()
                 - W * (1.0f - cos(d)) / d2
                 + W * W * (d - sin(d)) / (d2 * d);
    }
}

// ---------------------------------------------------------------------------
// Preintegrated::Preintegrated (main constructor)
// ---------------------------------------------------------------------------
// Initializes the preintegration object with:
//   b_    : the current best bias estimate (from previous optimization)
//   calib : IMU calibration (noise covariances Nga and NgaWalk)
//
// Nga     = measurement noise covariance = diag(ng², ng², ng², na², na², na²)
// NgaWalk = bias random walk covariance  = diag(ngw²,ngw²,ngw²,naw²,naw²,naw²)
// ---------------------------------------------------------------------------
Preintegrated::Preintegrated(const Bias &b_, const Calib &calib)
{
    Nga     = calib.Cov;      // Measurement noise covariance (from Allan variance)
    NgaWalk = calib.CovWalk;  // Bias random walk covariance  (from Allan variance)
    Initialize(b_);           // Reset all delta quantities to zero/identity
}

// Copy constructor — duplicates all state from an existing Preintegrated object
Preintegrated::Preintegrated(Preintegrated *pImuPre)
    : dT(pImuPre->dT), C(pImuPre->C), Info(pImuPre->Info),
      Nga(pImuPre->Nga), NgaWalk(pImuPre->NgaWalk), b(pImuPre->b),
      dR(pImuPre->dR), dV(pImuPre->dV), dP(pImuPre->dP),
      JRg(pImuPre->JRg), JVg(pImuPre->JVg), JVa(pImuPre->JVa),
      JPg(pImuPre->JPg), JPa(pImuPre->JPa),
      avgA(pImuPre->avgA), avgW(pImuPre->avgW),
      bu(pImuPre->bu), db(pImuPre->db),
      mvMeasurements(pImuPre->mvMeasurements)
{}

void Preintegrated::CopyFrom(Preintegrated *pImuPre)
{
    dT   = pImuPre->dT;
    C    = pImuPre->C;
    Info = pImuPre->Info;
    Nga     = pImuPre->Nga;
    NgaWalk = pImuPre->NgaWalk;
    b.CopyFrom(pImuPre->b);
    dR = pImuPre->dR;
    dV = pImuPre->dV;
    dP = pImuPre->dP;
    JRg = pImuPre->JRg;
    JVg = pImuPre->JVg;
    JVa = pImuPre->JVa;
    JPg = pImuPre->JPg;
    JPa = pImuPre->JPa;
    avgA = pImuPre->avgA;
    avgW = pImuPre->avgW;
    bu.CopyFrom(pImuPre->bu);
    db = pImuPre->db;
    mvMeasurements = pImuPre->mvMeasurements;
}

// ---------------------------------------------------------------------------
// Preintegrated::Initialize
// ---------------------------------------------------------------------------
// Resets all preintegrated quantities to their initial values.
// Called at the beginning of a new keyframe window, or before Reintegrate().
// ---------------------------------------------------------------------------
void Preintegrated::Initialize(const Bias &b_)
{
    dR.setIdentity();    // No rotation yet:    dR = I
    dV.setZero();        // No velocity yet:    dV = 0
    dP.setZero();        // No position yet:    dP = 0
    JRg.setZero();       // All bias Jacobians start at zero
    JVg.setZero();
    JVa.setZero();
    JPg.setZero();
    JPa.setZero();
    C.setZero();         // Zero covariance at start (no uncertainty yet)
    Info.setZero();
    db.setZero();        // Zero bias delta (b and bu are the same initially)
    b  = b_;             // Store the linearization bias
    bu = b_;             // Updated bias starts equal to linearization bias
    avgA.setZero();
    avgW.setZero();
    dT = 0.0f;           // No time has passed yet
    mvMeasurements.clear();  // Empty measurement buffer
}

// ---------------------------------------------------------------------------
// Preintegrated::Reintegrate
// ---------------------------------------------------------------------------
// When the optimizer produces a significantly improved bias estimate (bu),
// the first-order correction (via Jacobians) may not be accurate enough.
// In that case, this function re-runs the full integration from scratch
// using all stored raw measurements but with the updated bias bu.
//
// This is more expensive than bias correction but more accurate for large
// bias changes. ORB-SLAM3 calls this after bias optimization convergence.
// ---------------------------------------------------------------------------
void Preintegrated::Reintegrate()
{
    std::unique_lock<std::mutex> lock(mMutex);
    const std::vector<integrable> aux = mvMeasurements;  // copy raw measurements
    Initialize(bu);  // reset with updated bias bu as new linearization point
    for (size_t i = 0; i < aux.size(); i++)
        IntegrateNewMeasurement(aux[i].a, aux[i].w, aux[i].t);
}

// ---------------------------------------------------------------------------
// Preintegrated::IntegrateNewMeasurement
// ---------------------------------------------------------------------------
// THE CORE FUNCTION. Called once per IMU sample to accumulate measurements.
//
// Updates in this ORDER (important — each depends on the PREVIOUS step's dR):
//   1. dP  (position)  — depends on current dV and current dR (before update)
//   2. dV  (velocity)  — depends on current dR (before update)
//   3. dR  (rotation)  — updated last
//
// Also propagates:
//   - Covariance matrix C (15x15)
//   - Bias Jacobians JRg, JVg, JVa, JPg, JPa
//
// INPUTS:
//   acceleration : raw accel [ax, ay, az]  m/s²  in body frame
//   angVel       : raw gyro  [wx, wy, wz]  rad/s in body frame
//   dt           : timestep since last measurement (seconds)
// ---------------------------------------------------------------------------
void Preintegrated::IntegrateNewMeasurement(const Eigen::Vector3f &acceleration,
                                              const Eigen::Vector3f &angVel,
                                              const float &dt)
{
    // Store the raw measurement for potential future Reintegrate() calls
    mvMeasurements.push_back(integrable(acceleration, angVel, dt));

    // -----------------------------------------------------------------------
    // State transition matrix A (9x9) and noise input matrix B (9x6)
    // These relate the current state [dR, dV, dP] and its covariance
    // to the next state after integrating this measurement.
    //
    // State vector:  x = [dR(3), dV(3), dP(3)]   (9 elements)
    // Noise vector:  n = [n_gyro(3), n_accel(3)]  (6 elements)
    //
    // x_{k+1} = A * x_k + B * n_k
    // C_{k+1} = A * C_k * A^T + B * Nga * B^T
    // -----------------------------------------------------------------------
    Eigen::Matrix<float, 9, 9> A;
    A.setIdentity();
    Eigen::Matrix<float, 9, 6> B;
    B.setZero();

    // -----------------------------------------------------------------------
    // Bias-corrected measurements
    // acc  = true acceleration (body frame) = raw_accel - accel_bias
    // accW = true angular velocity           = raw_gyro  - gyro_bias
    // -----------------------------------------------------------------------
    Eigen::Vector3f acc, accW;
    acc  << acceleration(0) - b.bax,
            acceleration(1) - b.bay,
            acceleration(2) - b.baz;
    accW << angVel(0) - b.bwx,
            angVel(1) - b.bwy,
            angVel(2) - b.bwz;

    // -----------------------------------------------------------------------
    // Update running weighted averages of accel and gyro
    // These are used later for gravity direction initialization.
    //   avgA is accumulated in world frame (rotated by dR)
    //   avgW is accumulated in body frame
    // Weight is proportional to time: longer intervals contribute more
    // -----------------------------------------------------------------------
    avgA = (dT * avgA + dR * acc * dt) / (dT + dt);
    avgW = (dT * avgW + accW * dt)     / (dT + dt);

    // -----------------------------------------------------------------------
    // Step 1: Update delta POSITION dP
    // Uses the CURRENT (not yet updated) dR and dV.
    //
    // Continuous kinematic equation integrated over dt:
    //   p(t+dt) = p(t) + v(t)*dt + 0.5*R(t)*a*dt²
    //
    // In preintegration form (body frame of keyframe i):
    //   dP_{k+1} = dP_k + dV_k * dt + 0.5 * dR_k * acc * dt²
    // -----------------------------------------------------------------------
    dP = dP + dV * dt + 0.5f * dR * acc * dt * dt;

    // -----------------------------------------------------------------------
    // Step 2: Update delta VELOCITY dV
    // Uses the CURRENT (not yet updated) dR.
    //
    //   dV_{k+1} = dV_k + dR_k * acc * dt
    // -----------------------------------------------------------------------
    dV = dV + dR * acc * dt;

    // -----------------------------------------------------------------------
    // Skew-symmetric matrix of bias-corrected acceleration
    // Used in the A and B matrices for covariance propagation and Jacobians.
    // hat(acc) = [  0   -az   ay ]
    //            [  az   0   -ax ]
    //            [ -ay   ax   0  ]
    // -----------------------------------------------------------------------
    Eigen::Matrix<float, 3, 3> Wacc = Sophus::SO3f::hat(acc);

    // -----------------------------------------------------------------------
    // Fill A matrix blocks (state transition, using PRE-update dR)
    //
    // A describes how the state [dR, dV, dP] evolves:
    //
    //   A[3:6, 0:3] = -dR * dt * hat(acc)
    //     — How a rotation error in dR creates a velocity error in dV
    //       (because acc is rotated by dR before being integrated into dV)
    //
    //   A[6:9, 0:3] = -0.5 * dR * dt² * hat(acc)
    //     — How a rotation error in dR creates a position error in dP
    //       (same reasoning, quadratic in dt)
    //
    //   A[6:9, 3:6] = I * dt
    //     — How a velocity error in dV creates a position error in dP
    //       (dP = dP + dV * dt)
    //
    // Note: A[0:3, 0:3] = dRi.deltaR^T is filled AFTER computing dRi below.
    // -----------------------------------------------------------------------
    A.block<3, 3>(3, 0) = -dR * dt * Wacc;
    A.block<3, 3>(6, 0) = -0.5f * dR * dt * dt * Wacc;
    A.block<3, 3>(6, 3) =  Eigen::DiagonalMatrix<float, 3>(dt, dt, dt);

    // -----------------------------------------------------------------------
    // Fill B matrix blocks (noise input, using PRE-update dR)
    //
    //   B[3:6, 3:6] = dR * dt
    //     — How accelerometer noise maps to velocity uncertainty
    //
    //   B[6:9, 3:6] = 0.5 * dR * dt²
    //     — How accelerometer noise maps to position uncertainty
    //
    // Note: B[0:3, 0:3] = dRi.rightJ * dt is filled AFTER computing dRi below.
    // -----------------------------------------------------------------------
    B.block<3, 3>(3, 3) = dR * dt;
    B.block<3, 3>(6, 3) = 0.5f * dR * dt * dt;

    // -----------------------------------------------------------------------
    // Update bias Jacobians for POSITION and VELOCITY
    // (must be done BEFORE updating dR, as they use the current dR)
    //
    // These Jacobians answer: "if the bias changes by a small amount,
    // how much does dP/dV change?" — allowing bias correction without
    // full re-integration.
    //
    // Recursive update equations:
    //
    //   JPa_{k+1} = JPa_k + JVa_k * dt - 0.5 * dR_k * dt²
    //     — Position Jacobian w.r.t. accel bias:
    //       new dP depends on old dP (via JPa) + old dV (via JVa*dt)
    //       + direct accel term (-0.5 * dR * dt²)
    //
    //   JPg_{k+1} = JPg_k + JVg_k * dt - 0.5 * dR_k * dt² * hat(acc) * JRg_k
    //     — Position Jacobian w.r.t. gyro bias:
    //       same as JPa but gyro bias also affects dR (via JRg),
    //       which in turn affects how acc is rotated into world frame
    //
    //   JVa_{k+1} = JVa_k - dR_k * dt
    //     — Velocity Jacobian w.r.t. accel bias:
    //       direct: changing accel bias changes the integrated velocity
    //
    //   JVg_{k+1} = JVg_k - dR_k * dt * hat(acc) * JRg_k
    //     — Velocity Jacobian w.r.t. gyro bias:
    //       indirect: gyro bias changes dR (via JRg), which changes
    //       how acc is projected, which changes dV
    // -----------------------------------------------------------------------
    JPa = JPa + JVa * dt - 0.5f * dR * dt * dt;
    JPg = JPg + JVg * dt - 0.5f * dR * dt * dt * Wacc * JRg;
    JVa = JVa - dR * dt;
    JVg = JVg - dR * dt * Wacc * JRg;

    // -----------------------------------------------------------------------
    // Step 3: Update delta ROTATION dR
    //
    // Create dRi = Exp((angVel - b_gyro) * dt) — the rotation increment
    // for this single IMU step (using Rodrigues formula, see IntegratedRotation)
    //
    // Then accumulate: dR_{k+1} = dR_k * dRi
    // NormalizeRotation() projects back onto SO(3) to prevent numerical drift
    // -----------------------------------------------------------------------
    IntegratedRotation dRi(angVel, b, dt);
    dR = NormalizeRotation(dR * dRi.deltaR);

    // -----------------------------------------------------------------------
    // Fill remaining A and B blocks (now that dRi is available)
    //
    //   A[0:3, 0:3] = dRi.deltaR^T
    //     — How current rotation error propagates to next rotation error.
    //       Transpose because: if dR has a small error eps,
    //       then dR_new = dR * dRi has error dRi^T * eps * dRi ≈ dRi^T * eps
    //
    //   B[0:3, 0:3] = dRi.rightJ * dt
    //     — How gyro noise maps to rotation uncertainty.
    //       rightJ accounts for the nonlinearity of the exponential map:
    //       noise in the rotation vector phi maps to rotation matrix noise
    //       via the Right Jacobian Jr(phi)
    // -----------------------------------------------------------------------
    A.block<3, 3>(0, 0) = dRi.deltaR.transpose();
    B.block<3, 3>(0, 0) = dRi.rightJ * dt;

    // -----------------------------------------------------------------------
    // Update covariance matrix C (15x15)
    //
    // The 15x15 C matrix covers [dR(3), dV(3), dP(3), b_gyro(3), b_accel(3)].
    //
    // Upper-left 9x9 block — propagate [dR, dV, dP] covariance:
    //   C[0:9, 0:9] = A * C[0:9, 0:9] * A^T + B * Nga * B^T
    //
    //   This is the standard linear covariance propagation formula:
    //     - A * C * A^T : how existing uncertainty propagates through dynamics
    //     - B * Nga * B^T : new uncertainty added by this measurement's noise
    //                       Nga = diag(ng², ng², ng², na², na², na²)
    //
    // Lower-right 6x6 block — propagate bias covariance:
    //   C[9:15, 9:15] += NgaWalk
    //
    //   The bias random walk adds NgaWalk = diag(ngw², ngw², ngw², naw², naw², naw²)
    //   at every step. This models the fact that gyro/accel bias slowly drifts
    //   over time (modeled as a Wiener process / random walk).
    //   Note: "+=" because bias uncertainty accumulates — it never decreases
    //   from integration alone (only the optimizer can reduce it).
    // -----------------------------------------------------------------------
    C.block<9, 9>(0, 0) = A * C.block<9, 9>(0, 0) * A.transpose()
                          + B * Nga * B.transpose();
    C.block<6, 6>(9, 9) += NgaWalk;

    // -----------------------------------------------------------------------
    // Update rotation bias Jacobian JRg
    //
    // JRg answers: "if the gyro bias changes by delta_bg, how much does dR change?"
    //
    // Recursive update:
    //   JRg_{k+1} = dRi.deltaR^T * JRg_k - dRi.rightJ * dt
    //
    // Intuition:
    //   - dRi.deltaR^T * JRg_k :
    //       The previous Jacobian, rotated into the new frame by dRi^T.
    //       Because dR = dR * dRi, a change in dR at step k propagates
    //       through dRi^T to affect dR at step k+1.
    //
    //   - dRi.rightJ * dt :
    //       The direct contribution: changing bias by delta_bg changes
    //       angVel by -delta_bg, which changes the rotation vector phi
    //       by -delta_bg * dt, which changes deltaR by Jr * (-delta_bg * dt).
    //       The minus sign means increasing bias decreases the integrated rotation.
    // -----------------------------------------------------------------------
    JRg = dRi.deltaR.transpose() * JRg - dRi.rightJ * dt;

    // Accumulate total integration time
    dT += dt;
}

// ---------------------------------------------------------------------------
// Preintegrated::MergePrevious
// ---------------------------------------------------------------------------
// Merges a previous preintegration window (pPrev) into this one.
// Used when two consecutive keyframes are merged into one during loop closure
// or keyframe culling — their IMU windows must be combined.
//
// Process:
//   1. Take all raw measurements from pPrev, then from this window
//   2. Re-integrate everything from scratch with the current updated bias bu
// ---------------------------------------------------------------------------
void Preintegrated::MergePrevious(Preintegrated *pPrev)
{
    if (pPrev == this) return;  // Nothing to merge if same object

    std::unique_lock<std::mutex> lock1(mMutex);
    std::unique_lock<std::mutex> lock2(pPrev->mMutex);

    // Use the current updated bias as the new linearization point
    Bias bav;
    bav.bwx = bu.bwx; bav.bwy = bu.bwy; bav.bwz = bu.bwz;
    bav.bax = bu.bax; bav.bay = bu.bay; bav.baz = bu.baz;

    // Copy both measurement vectors
    const std::vector<integrable> aux1 = pPrev->mvMeasurements;
    const std::vector<integrable> aux2 = mvMeasurements;

    // Re-integrate: first previous window, then current window
    Initialize(bav);
    for (size_t i = 0; i < aux1.size(); i++)
        IntegrateNewMeasurement(aux1[i].a, aux1[i].w, aux1[i].t);
    for (size_t i = 0; i < aux2.size(); i++)
        IntegrateNewMeasurement(aux2[i].a, aux2[i].w, aux2[i].t);
}

// ---------------------------------------------------------------------------
// Preintegrated::SetNewBias
// ---------------------------------------------------------------------------
// Called by the optimizer when it produces an updated bias estimate bu_.
// Stores bu_ and computes the delta bias db = bu_ - b.
//
// db is then used by GetUpdatedDelta*() functions to apply first-order
// bias corrections to dR/dV/dP without re-integration.
// ---------------------------------------------------------------------------
void Preintegrated::SetNewBias(const Bias &bu_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    bu = bu_;
    // db = [delta_bw, delta_ba] = updated_bias - original_bias
    db(0) = bu_.bwx - b.bwx;
    db(1) = bu_.bwy - b.bwy;
    db(2) = bu_.bwz - b.bwz;
    db(3) = bu_.bax - b.bax;
    db(4) = bu_.bay - b.bay;
    db(5) = bu_.baz - b.baz;
}

// ---------------------------------------------------------------------------
// GetDeltaBias — returns the difference between a given bias and original bias
// ---------------------------------------------------------------------------
IMU::Bias Preintegrated::GetDeltaBias(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    return IMU::Bias(b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz,
                     b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz);
}

// ---------------------------------------------------------------------------
// GetDeltaRotation (with explicit bias b_)
// ---------------------------------------------------------------------------
// First-order bias-corrected delta rotation for a given bias b_.
//
// Corrected dR = dR * Exp(JRg * (b_.bwxyz - b.bwxyz))
//
// Exp() here is the SO(3) exponential map (Rodrigues formula).
// JRg * dbg gives the rotation vector perturbation due to the bias change.
// This is valid when (b_ - b) is small — if large, use Reintegrate().
// ---------------------------------------------------------------------------
Eigen::Matrix3f Preintegrated::GetDeltaRotation(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3f dbg;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;
    // Apply first-order correction: dR_corrected = dR * Exp(JRg * dbg)
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * dbg).matrix());
}

// ---------------------------------------------------------------------------
// GetDeltaVelocity (with explicit bias b_)
// ---------------------------------------------------------------------------
// First-order bias-corrected delta velocity:
//   dV_corrected = dV + JVg * dbg + JVa * dba
// ---------------------------------------------------------------------------
Eigen::Vector3f Preintegrated::GetDeltaVelocity(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3f dbg, dba;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;
    dba << b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz;
    return dV + JVg * dbg + JVa * dba;
}

// ---------------------------------------------------------------------------
// GetDeltaPosition (with explicit bias b_)
// ---------------------------------------------------------------------------
// First-order bias-corrected delta position:
//   dP_corrected = dP + JPg * dbg + JPa * dba
// ---------------------------------------------------------------------------
Eigen::Vector3f Preintegrated::GetDeltaPosition(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3f dbg, dba;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;
    dba << b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz;
    return dP + JPg * dbg + JPa * dba;
}

// ---------------------------------------------------------------------------
// GetUpdated* variants — same as above but use internally stored db = bu - b
// ---------------------------------------------------------------------------

// db.head(3) = delta gyro bias  [dbwx, dbwy, dbwz]
// db.tail(3) = delta accel bias [dbax, dbay, dbaz]

Eigen::Matrix3f Preintegrated::GetUpdatedDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * db.head(3)).matrix());
}

Eigen::Vector3f Preintegrated::GetUpdatedDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV + JVg * db.head(3) + JVa * db.tail(3);
}

Eigen::Vector3f Preintegrated::GetUpdatedDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP + JPg * db.head(3) + JPa * db.tail(3);
}

// ---------------------------------------------------------------------------
// GetOriginal* variants — raw values without any bias correction
// ---------------------------------------------------------------------------
Eigen::Matrix3f Preintegrated::GetOriginalDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dR;
}

Eigen::Vector3f Preintegrated::GetOriginalDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV;
}

Eigen::Vector3f Preintegrated::GetOriginalDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP;
}

Bias Preintegrated::GetOriginalBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return b;
}

Bias Preintegrated::GetUpdatedBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return bu;
}

Eigen::Matrix<float, 6, 1> Preintegrated::GetDeltaBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return db;
}

// ---------------------------------------------------------------------------
// Bias::CopyFrom — simple field copy
// ---------------------------------------------------------------------------
void Bias::CopyFrom(Bias &b)
{
    bax = b.bax; bay = b.bay; baz = b.baz;
    bwx = b.bwx; bwy = b.bwy; bwz = b.bwz;
}

// Stream operator for printing bias values
std::ostream &operator<<(std::ostream &out, const Bias &b)
{
    if (b.bwx > 0) out << " "; out << b.bwx << ",";
    if (b.bwy > 0) out << " "; out << b.bwy << ",";
    if (b.bwz > 0) out << " "; out << b.bwz << ",";
    if (b.bax > 0) out << " "; out << b.bax << ",";
    if (b.bay > 0) out << " "; out << b.bay << ",";
    if (b.baz > 0) out << " "; out << b.baz;
    return out;
}

// ---------------------------------------------------------------------------
// Calib::Set
// ---------------------------------------------------------------------------
// Stores the extrinsic transform and builds the noise covariance matrices.
//
// The noise parameters are squared because we store variance (σ²), not σ:
//   ng²  = gyro noise density squared    → diagonal of Cov     [0:3]
//   na²  = accel noise density squared   → diagonal of Cov     [3:6]
//   ngw² = gyro random walk squared      → diagonal of CovWalk [0:3]
//   naw² = accel random walk squared     → diagonal of CovWalk [3:6]
//
// These come directly from your Allan variance test results.
// ---------------------------------------------------------------------------
void Calib::Set(const Sophus::SE3<float> &sophTbc,
                const float &ng, const float &na,
                const float &ngw, const float &naw)
{
    mbIsSet = true;
    const float ng2  = ng  * ng;   // gyro noise density²
    const float na2  = na  * na;   // accel noise density²
    const float ngw2 = ngw * ngw;  // gyro random walk²
    const float naw2 = naw * naw;  // accel random walk²

    mTbc = sophTbc;          // IMU body -> camera extrinsic transform
    mTcb = mTbc.inverse();   // camera -> IMU body (inverse)

    // Nga: measurement noise, diagonal [ng², ng², ng², na², na², na²]
    Cov.diagonal() << ng2, ng2, ng2, na2, na2, na2;

    // NgaWalk: bias random walk, diagonal [ngw², ngw², ngw², naw², naw², naw²]
    CovWalk.diagonal() << ngw2, ngw2, ngw2, naw2, naw2, naw2;
}

Calib::Calib(const Calib &calib)
{
    mbIsSet = calib.mbIsSet;
    mTbc    = calib.mTbc;
    mTcb    = calib.mTcb;
    Cov     = calib.Cov;
    CovWalk = calib.CovWalk;
}

} // namespace IMU
} // namespace ORB_SLAM3