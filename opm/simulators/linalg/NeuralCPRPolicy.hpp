/*
  Copyright 2024 OPM Contributors.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_NEURAL_CPR_POLICY_HPP
#define OPM_NEURAL_CPR_POLICY_HPP

#include <opm/simulators/linalg/NeuralCPRFeatures.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>

namespace Opm {

/// The three axes of CPR solver configuration the policy controls.
enum class CPRDecoupling    { QuasiIMPES, TrueIMPES, TrueIMPESAnalytic };
enum class CPRFineSmoother  { ParOverILU0, DILU };
enum class CPRCoarseSmoother{ ILU0, DILU };

struct CPRConfig {
    CPRDecoupling    decoupling = CPRDecoupling::TrueIMPES;
    bool             use_cprw  = true;
    CPRFineSmoother  fine      = CPRFineSmoother::DILU;
    CPRCoarseSmoother coarse   = CPRCoarseSmoother::ILU0;

    bool operator==(const CPRConfig& o) const noexcept
    {
        return decoupling == o.decoupling && use_cprw == o.use_cprw
               && fine == o.fine && coarse == o.coarse;
    }
    bool operator!=(const CPRConfig& o) const noexcept { return !(*this == o); }
};

/// Two-hidden-layer MLP (8→32→16→8 logits) that maps solver-state features to
/// a CPR configuration.  Weights are loaded from a flat binary file written by
/// the companion Python training script.  When no weights file is available the
/// policy falls back to a lightweight rule-based heuristic so the hook can be
/// exercised immediately without trained weights.
///
/// Binary weight file layout (all IEEE-754 float32, little-endian):
///   magic    : uint32  = 0x4E435052  ('NCPR')
///   version  : uint32  = 1
///   W1[32][8]: 256 f32  (row-major)
///   b1[32]   : 32  f32
///   W2[16][32]: 512 f32  (row-major)
///   b2[16]   : 16  f32
///   W3[8][16] : 128 f32  (row-major)
///   b3[8]    : 8   f32
///   Total    : 952 floats = 3816 bytes (including 8-byte header)
class NeuralCPRPolicy {
public:
    /// Construct.  If weights_path is empty or the file cannot be opened the
    /// instance falls back to rule-based prediction (isLoaded() == false).
    explicit NeuralCPRPolicy(const std::string& weights_path = "")
    {
        if (!weights_path.empty())
            loadWeights(weights_path);
    }

    bool isLoaded() const { return loaded_; }

    CPRConfig predict(const NeuralCPRFeatures& feat) const
    {
        if (!loaded_)
            return ruleBasedPredict(feat);

        auto x = feat.toInputVector();
        return mlpPredict(x);
    }

    /// Patch only the fields that differ between prev and cfg into an existing
    /// PropertyTree, avoiding the 30+ put() calls of a full rebuild.
    /// Precondition: cfg.use_cprw == prev.use_cprw (same preconditioner type).
    static void updatePropertyTreeInPlace(const CPRConfig& cfg,
                                          const CPRConfig& prev,
                                          PropertyTree& prm)
    {
        if (cfg.decoupling != prev.decoupling)
            prm.put("preconditioner.weight_type", decouplingStr(cfg.decoupling));
        if (cfg.fine != prev.fine)
            prm.put("preconditioner.finesmoother.type", fineStr(cfg.fine));
        if (cfg.coarse != prev.coarse)
            prm.put("preconditioner.coarsesolver.preconditioner.smoother",
                    coarseStr(cfg.coarse));
    }

    /// Build an Opm::PropertyTree for the chosen CPRConfig, mirroring
    /// setupCPRW() in setupPropertyTree.cpp so the tree is accepted by
    /// FlexibleSolver without modification.
    static PropertyTree configToPropertyTree(const CPRConfig& cfg,
                                             double tol     = 0.005,
                                             int    maxiter = 20)
    {
        using namespace std::string_literals;
        PropertyTree prm;
        prm.put("maxiter",   maxiter);
        prm.put("tol",       tol);
        prm.put("verbosity", 0);
        prm.put("solver",    "bicgstab"s);

        prm.put("preconditioner.type",             cfg.use_cprw ? "cprw"s : "cpr"s);
        prm.put("preconditioner.use_well_weights", cfg.use_cprw ? "true"s : "false"s);
        prm.put("preconditioner.add_wells",        cfg.use_cprw ? "true"s : "false"s);
        prm.put("preconditioner.weight_type",      decouplingStr(cfg.decoupling));
        prm.put("preconditioner.pre_smooth",       0);
        prm.put("preconditioner.post_smooth",      1);
        prm.put("preconditioner.finesmoother.type",       fineStr(cfg.fine));
        prm.put("preconditioner.finesmoother.relaxation", 1.0);
        prm.put("preconditioner.verbosity",        0);
        prm.put("preconditioner.coarsesolver.maxiter",   1);
        prm.put("preconditioner.coarsesolver.tol",       1e-1);
        prm.put("preconditioner.coarsesolver.solver",    "loopsolver"s);
        prm.put("preconditioner.coarsesolver.verbosity", 0);

        const std::string amgRoot = "preconditioner.coarsesolver.preconditioner.";
        prm.put(amgRoot + "type",             "amg"s);
        prm.put(amgRoot + "smoother",         coarseStr(cfg.coarse));
        prm.put(amgRoot + "alpha",            0.333333333333);
        prm.put(amgRoot + "relaxation",       1.0);
        prm.put(amgRoot + "iterations",       1);
        prm.put(amgRoot + "coarsenTarget",    1200);
        prm.put(amgRoot + "pre_smooth",       1);
        prm.put(amgRoot + "post_smooth",      1);
        prm.put(amgRoot + "beta",             0.0);
        prm.put(amgRoot + "verbosity",        0);
        prm.put(amgRoot + "maxlevel",         15);
        prm.put(amgRoot + "skip_isolated",    0);
        prm.put(amgRoot + "accumulate",       1);
        prm.put(amgRoot + "prolongationdamping", 1.0);
        prm.put(amgRoot + "maxdistance",      2);
        prm.put(amgRoot + "maxconnectivity",  15);
        prm.put(amgRoot + "maxaggsize",       6);
        prm.put(amgRoot + "minaggsize",       4);
        return prm;
    }

private:
    // ---- Network weights ----
    std::array<std::array<float, 8>,  32> W1_{};
    std::array<float, 32>                 b1_{};
    std::array<std::array<float, 32>, 16> W2_{};
    std::array<float, 16>                 b2_{};
    std::array<std::array<float, 16>,  8> W3_{};
    std::array<float, 8>                  b3_{};
    bool loaded_ = false;

    static constexpr uint32_t kMagic   = 0x4E435052u;
    static constexpr uint32_t kVersion = 1u;

    void loadWeights(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return;

        uint32_t magic = 0, version = 0;
        f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != kMagic || version != kVersion)
            return;

        auto readFloats = [&](float* dst, std::size_t n) {
            f.read(reinterpret_cast<char*>(dst), n * sizeof(float));
        };

        readFloats(W1_[0].data(), 32 * 8);
        readFloats(b1_.data(),    32);
        readFloats(W2_[0].data(), 16 * 32);
        readFloats(b2_.data(),    16);
        readFloats(W3_[0].data(),  8 * 16);
        readFloats(b3_.data(),     8);

        if (f.good())
            loaded_ = true;
    }

    CPRConfig mlpPredict(const std::array<float, 8>& x) const
    {
        // Layer 1: h1 = ReLU(W1 x + b1),  shape [32]
        std::array<float, 32> h1;
        for (int i = 0; i < 32; ++i) {
            float v = b1_[i];
            for (int j = 0; j < 8; ++j)
                v += W1_[i][j] * x[j];
            h1[i] = relu(v);
        }

        // Layer 2: h2 = ReLU(W2 h1 + b2),  shape [16]
        std::array<float, 16> h2;
        for (int i = 0; i < 16; ++i) {
            float v = b2_[i];
            for (int j = 0; j < 32; ++j)
                v += W2_[i][j] * h1[j];
            h2[i] = relu(v);
        }

        // Layer 3: logits = W3 h2 + b3,  shape [8]
        std::array<float, 8> logits;
        for (int i = 0; i < 8; ++i) {
            float v = b3_[i];
            for (int j = 0; j < 16; ++j)
                v += W3_[i][j] * h2[j];
            logits[i] = v;
        }

        // Decode logit groups:
        // [0:3] → decoupling (3-way softmax argmax)
        // [3]   → use_cprw  (sigmoid > 0)
        // [4:6] → fine smoother (2-way)
        // [6:8] → coarse smoother (2-way)
        CPRConfig cfg;
        int dec = argmax(logits.data(), 3);
        if      (dec == 0) cfg.decoupling = CPRDecoupling::QuasiIMPES;
        else if (dec == 1) cfg.decoupling = CPRDecoupling::TrueIMPES;
        else               cfg.decoupling = CPRDecoupling::TrueIMPESAnalytic;

        cfg.use_cprw = (logits[3] > 0.0f);
        cfg.fine     = (logits[4] >= logits[5]) ? CPRFineSmoother::ParOverILU0
                                                 : CPRFineSmoother::DILU;
        cfg.coarse   = (logits[6] >= logits[7]) ? CPRCoarseSmoother::ILU0
                                                 : CPRCoarseSmoother::DILU;
        return cfg;
    }

    /// Rule-based heuristic used when no weights are loaded.
    ///
    /// Strategy:
    ///  - First Newton step (iter 0): quasiimpes + ParOverILU0.
    ///    Weight computation is O(n) instead of O(n×numEq³), and the fine
    ///    smoother setup is cheaper.  Quality is sufficient at the start of a
    ///    timestep because the initial residual reduction ratio is not yet known.
    ///  - Slow convergence (residual barely moving) or many iterations:
    ///    trueimpes + DILU — maximises robustness.
    ///  - Otherwise: quasiimpes + DILU — balanced cost/quality.
    static CPRConfig ruleBasedPredict(const NeuralCPRFeatures& f)
    {
        CPRConfig cfg;
        cfg.use_cprw = true;
        cfg.coarse   = CPRCoarseSmoother::ILU0;

        const bool slowConv = (f.nl_iteration > 0 && f.nl_residual_reduce > 0.5);
        const bool manyIter = (f.nl_iteration >= 4);

        if (slowConv || manyIter) {
            cfg.decoupling = CPRDecoupling::TrueIMPES;
            cfg.fine       = CPRFineSmoother::DILU;
        } else if (f.nl_iteration == 0) {
            cfg.decoupling = CPRDecoupling::QuasiIMPES;
            cfg.fine       = CPRFineSmoother::ParOverILU0;
        } else {
            cfg.decoupling = CPRDecoupling::QuasiIMPES;
            cfg.fine       = CPRFineSmoother::DILU;
        }
        return cfg;
    }

    static float relu(float x) { return x > 0.0f ? x : 0.0f; }

    static int argmax(const float* v, int n)
    {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (v[i] > v[best]) best = i;
        return best;
    }

    static std::string decouplingStr(CPRDecoupling d)
    {
        switch (d) {
            case CPRDecoupling::QuasiIMPES:        return "quasiimpes";
            case CPRDecoupling::TrueIMPES:         return "trueimpes";
            case CPRDecoupling::TrueIMPESAnalytic: return "trueimpesanalytic";
        }
        return "trueimpes";
    }

    static std::string fineStr(CPRFineSmoother s)
    {
        return s == CPRFineSmoother::DILU ? "dilu" : "paroverilu0";
    }

    static std::string coarseStr(CPRCoarseSmoother s)
    {
        return s == CPRCoarseSmoother::DILU ? "dilu" : "ilu0";
    }
};

} // namespace Opm

#endif // OPM_NEURAL_CPR_POLICY_HPP
