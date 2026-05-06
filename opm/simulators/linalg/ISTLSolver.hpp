/*
  Copyright 2016 IRIS AS
  Copyright 2019, 2020 Equinor ASA
  Copyright 2020 SINTEF Digital, Mathematics and Cybernetics

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

#ifndef OPM_ISTLSOLVER_HEADER_INCLUDED
#define OPM_ISTLSOLVER_HEADER_INCLUDED

#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/solver.hh>

#include <opm/common/CriticalError.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/Exceptions.hpp>
#include <opm/common/TimingMacros.hpp>

#include <opm/grid/utility/ElementChunks.hpp>

#include <opm/models/discretization/common/fvbaseproperties.hh>
#include <opm/models/common/multiphasebaseproperties.hh>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>
#include <opm/simulators/flow/BlackoilModelParameters.hpp>
#include <opm/simulators/flow/FlowBaseVanguard.hpp>
#include <opm/simulators/flow/FlowBaseProblemProperties.hpp>
#include <opm/simulators/linalg/ExtractParallelGridInformationToISTL.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/matrixblock.hh>
#include <opm/simulators/linalg/istlsparsematrixadapter.hh>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/WriteSystemMatrixHelper.hpp>
#include <opm/simulators/linalg/findOverlapRowsAndColumns.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>
#include <opm/simulators/linalg/AbstractISTLSolver.hpp>
#include <opm/simulators/linalg/printlinearsolverparameter.hpp>
#include <opm/simulators/linalg/CprPolicyAction.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>
#include <opm/simulators/linalg/CprPolicyFeatures.hpp>
#include <opm/simulators/linalg/NeuralCprPolicy.hpp>

#include <fmt/format.h>

#include <any>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace Opm::Properties {

namespace TTag {
struct FlowIstlSolver {
    using InheritsFrom = std::tuple<FlowIstlSolverParams>;
};
}

template <class TypeTag, class MyTypeTag>
struct WellModel;

//! Set the type of a global jacobian matrix for linear solvers that are based on
//! dune-istl.
template<class TypeTag>
struct SparseMatrixAdapter<TypeTag, TTag::FlowIstlSolver>
{
private:
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    enum { numEq = getPropValue<TypeTag, Properties::NumEq>() };
    using Block = MatrixBlock<Scalar, numEq, numEq>;

public:
    using type = typename Linear::IstlSparseMatrixAdapter<Block>;
};

} // namespace Opm::Properties

namespace Opm
{


namespace detail
{

template<class Matrix, class Vector, class Comm>
struct FlexibleSolverInfo
{
    using AbstractSolverType = Dune::InverseOperator<Vector, Vector>;
    using AbstractOperatorType = Dune::AssembledLinearOperator<Matrix, Vector, Vector>;
    using AbstractPreconditionerType = Dune::PreconditionerWithUpdate<Vector, Vector>;

    void create(const Matrix& matrix,
                bool parallel,
                const PropertyTree& prm,
                std::size_t pressureIndex,
                std::function<Vector()> weightCalculator,
                const bool forceSerial,
                Comm* comm);

    std::unique_ptr<AbstractSolverType> solver_;
    std::unique_ptr<AbstractOperatorType> op_;
    std::unique_ptr<LinearOperatorExtra<Vector,Vector>> wellOperator_;
    AbstractPreconditionerType* pre_ = nullptr;
    std::size_t interiorCellNum_ = 0;
};


#ifdef HAVE_MPI
/// Copy values in parallel.
void copyParValues(std::any& parallelInformation, std::size_t size,
                   Dune::OwnerOverlapCopyCommunication<int,int>& comm);
#endif

/// Zero out off-diagonal blocks on rows corresponding to overlap cells
/// Diagonal blocks on ovelap rows are set to diag(1.0).
template<class Matrix>
void makeOverlapRowsInvalid(Matrix& matrix,
                            const std::vector<int>& overlapRows);

/// Create sparsity pattern for block-Jacobi matrix based on partitioning of grid.
/// Do not initialize the values, that is done in copyMatToBlockJac()
template<class Matrix, class Grid>
std::unique_ptr<Matrix> blockJacobiAdjacency(const Grid& grid,
                                             const std::vector<int>& cell_part,
                                             std::size_t nonzeroes,
                                             const std::vector<std::set<int>>& wellConnectionsGraph);
}

    /// This class solves the fully implicit black-oil system by
    /// solving the reduced system (after eliminating well variables)
    /// as a block-structured matrix (one block for all cell variables) for a fixed
    /// number of cell variables np .
    template <class TypeTag>
    class ISTLSolver : public AbstractISTLSolver<GetPropType<TypeTag, Properties::SparseMatrixAdapter>,
                                                 GetPropType<TypeTag, Properties::GlobalEqVector>>
    {
    protected:
        using GridView = GetPropType<TypeTag, Properties::GridView>;
        using Scalar = GetPropType<TypeTag, Properties::Scalar>;
        using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;
        using Vector = GetPropType<TypeTag, Properties::GlobalEqVector>;
        using Indices = GetPropType<TypeTag, Properties::Indices>;
        using WellModel = GetPropType<TypeTag, Properties::WellModel>;
        using Simulator = GetPropType<TypeTag, Properties::Simulator>;
        using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
        using Matrix = typename SparseMatrixAdapter::IstlMatrix;
        using ThreadManager = GetPropType<TypeTag, Properties::ThreadManager>;
        using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
        using AbstractSolverType = Dune::InverseOperator<Vector, Vector>;
        using AbstractOperatorType = Dune::AssembledLinearOperator<Matrix, Vector, Vector>;
        using AbstractPreconditionerType = Dune::PreconditionerWithUpdate<Vector, Vector>;
        using WellModelOperator = WellModelAsLinearOperator<WellModel, Vector, Vector>;
        using ElementMapper = GetPropType<TypeTag, Properties::ElementMapper>;
        using ElementChunksType = ElementChunks<GridView, Dune::Partitions::All>;

        constexpr static std::size_t pressureIndex = GetPropType<TypeTag, Properties::Indices>::pressureSwitchIdx;

        static constexpr bool enablePolymerMolarWeight = getPropValue<TypeTag, Properties::EnablePolymerMW>();
        constexpr static bool isIncompatibleWithCprw = enablePolymerMolarWeight;

#if HAVE_MPI
        using CommunicationType = Dune::OwnerOverlapCopyCommunication<int,int>;
#else
        using CommunicationType = Dune::Communication<int>;
#endif

    public:
        using AssembledLinearOperatorType = Dune::AssembledLinearOperator< Matrix, Vector, Vector >;

        static void registerParameters()
        {
            FlowLinearSolverParameters::registerParameters();
        }

        /// Construct a system solver.
        /// \param[in] simulator   The opm-models simulator object
        /// \param[in] parameters  Explicit parameters for solver setup, do not
        ///                        read them from command line parameters.
        /// \param[in] forceSerial If true, will set up a serial linear solver only,
        ///                        local to the current rank, instead of creating a
        ///                        parallel (MPI distributed) linear solver.
        ISTLSolver(const Simulator& simulator,
                   const FlowLinearSolverParameters& parameters,
                   bool forceSerial = false)
            : simulator_(simulator),
              iterations_( 0 ),
              matrix_(nullptr),
              parameters_{parameters},
              forceSerial_(forceSerial)
        {
            initialize();
        }

        /// Construct a system solver.
        /// \param[in] simulator   The opm-models simulator object
        explicit ISTLSolver(const Simulator& simulator)
            : simulator_(simulator),
              iterations_( 0 ),
              solveCount_(0),
              matrix_(nullptr)
        {
            parameters_.resize(1);
            parameters_[0].init(simulator_.vanguard().eclState().getSimulationConfig().useCPR());
            initialize();
        }

        void initialize()
        {
            OPM_TIMEBLOCK(IstlSolver);

            if (isIncompatibleWithCprw) {
                // Polymer injectivity is incompatible with the CPRW linear solver.
                // Instead of aborting the run, emit a warning and fall back to ILU0
                // so that polymer injectivity cases work with default parameters.
                if (parameters_[0].linsolver_ == "cprw" || parameters_[0].linsolver_ == "hybrid") {
                    const bool on_io_rank = (simulator_.gridView().comm().rank() == 0);
                    const std::string msg =
                        fmt::format("The polymer injectivity model is incompatible with the '{}' "
                                    "linear solver. Falling back to --linear-solver=ilu0.",
                                    parameters_[0].linsolver_);
                    if (on_io_rank) {
                        OpmLog::warning(msg);
                    }
                    parameters_[0].linsolver_ = "ilu0";
                }
            }

            if (parameters_[0].linsolver_ == "hybrid") {
                // Experimental hybrid configuration.
                // When chosen, will set up two solvers, one with CPRW
                // and the other with ILU0 preconditioner. More general
                // options may be added later.
                prm_.clear();
                parameters_.clear();
                {
                    FlowLinearSolverParameters para;
                    para.init(false);
                    para.linsolver_ = "cprw";
                    parameters_.push_back(para);
                    prm_.push_back(setupPropertyTree(parameters_[0],
                                                     Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
                                                     Parameters::IsSet<Parameters::LinearSolverReduction>()));
                }
                {
                    FlowLinearSolverParameters para;
                    para.init(false);
                    para.linsolver_ = "ilu0";
                    parameters_.push_back(para);
                    prm_.push_back(setupPropertyTree(parameters_[1],
                                                     Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
                                                     Parameters::IsSet<Parameters::LinearSolverReduction>()));
                }
                // ------------
            } else {
                assert(parameters_.size() == 1);
                assert(prm_.empty());

                // Do a normal linear solver setup.
                if (parameters_[0].is_nldd_local_solver_) {
                    prm_.push_back(setupPropertyTree(parameters_[0],
                                                     Parameters::IsSet<Parameters::NlddLocalLinearSolverMaxIter>(),
                                                     Parameters::IsSet<Parameters::NlddLocalLinearSolverReduction>()));
                }
                else {
                    prm_.push_back(setupPropertyTree(parameters_[0],
                                                     Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
                                                     Parameters::IsSet<Parameters::LinearSolverReduction>()));
                }
            }
            flexibleSolver_.resize(prm_.size());

            const bool on_io_rank = (simulator_.gridView().comm().rank() == 0);
#if HAVE_MPI
            comm_.reset( new CommunicationType( simulator_.vanguard().grid().comm() ) );
#endif
            extractParallelGridInformationToISTL(simulator_.vanguard().grid(), parallelInformation_);

            // For some reason simulator_.model().elementMapper() is not initialized at this stage
            //const auto& elemMapper = simulator_.model().elementMapper(); //does not work.
            // Set it up manually
            ElementMapper elemMapper(simulator_.vanguard().gridView(), Dune::mcmgElementLayout());
            detail::findOverlapAndInterior(simulator_.vanguard().grid(), elemMapper, overlapRows_, interiorRows_);
            useWellConn_ = Parameters::Get<Parameters::MatrixAddWellContributions>();
            const bool ownersFirst = Parameters::Get<Parameters::OwnerCellsFirst>();
            if (!ownersFirst) {
                const std::string msg = "The linear solver no longer supports --owner-cells-first=false.";
                if (on_io_rank) {
                    OpmLog::error(msg);
                }
                OPM_THROW_NOLOG(std::runtime_error, msg);
            }

            const int interiorCellNum_ = detail::numMatrixRowsToUseInSolver(simulator_.vanguard().grid(), true);
            for (auto& f : flexibleSolver_) {
                f.interiorCellNum_ = interiorCellNum_;
            }

#if HAVE_MPI
            if (isParallel()) {
                const std::size_t size = simulator_.vanguard().grid().leafGridView().size(0);
                detail::copyParValues(parallelInformation_, size, *comm_);
            }
#endif

            // Print parameters to PRT/DBG logs.
            detail::printLinearSolverParameters(parameters_, activeSolverNum_, prm_,  simulator_.gridView().comm());

            element_chunks_ = std::make_unique<ElementChunksType>(simulator_.vanguard().gridView(), Dune::Partitions::all, ThreadManager::maxThreads());

            initNeuralPolicy();
        }

        // Drop the cached matrix pointer so initPrepare() sees the next matrix
        // as a new object.  Callers that rebuild the linear system mid-run rely
        // on this - opm-flowgeomechanics does it from FlowProblemMech when
        // fracture connections change.
        void eraseMatrix() override
        {
            matrix_ = nullptr;
            rhs_ = nullptr;
            iterations_ = 0;
            solveCount_ = 0;

            for (auto& solverInfo : flexibleSolver_) {
                solverInfo.pre_ = nullptr;
                solverInfo.solver_.reset();
                solverInfo.op_.reset();
            }
        }

        /// Load the neural CPR policy from $OPM_NEURAL_CPR_WEIGHTS (if set).
        /// Also opens the sweep log file if OPM_CPR_SWEEP_LOG is set.
        void initNeuralPolicy()
        {
            const char* path = std::getenv("OPM_NEURAL_CPR_WEIGHTS");
            neural_policy_ = std::make_unique<NeuralCprPolicy>(path ? path : "");
            const bool rank0 = (simulator_.gridView().comm().rank() == 0);

            // Minimum softmax probability required before switching configs.
            // Below threshold the current config is reused, avoiding rebuilds
            // triggered by low-confidence predictions.  Default 0.6.
            const char* thr_env = std::getenv("OPM_NEURAL_CPR_CONFIDENCE");
            confidence_threshold_ = thr_env ? std::stof(thr_env) : 0.6f;

            // OPM_NEURAL_CPR_FORBID_LABELS=a,b,c — comma-separated list of
            // label indices (0-23) to exclude from selection.  The policy picks
            // the next highest-scoring allowed label instead.
            // Example: "16,17,18,19,20,21,22,23" blocks all trueimpesanalytic
            // configs for decks where that weight type is unsupported.
            {
                const char* forbid_env = std::getenv("OPM_NEURAL_CPR_FORBID_LABELS");
                if (forbid_env) {
                    std::string s(forbid_env);
                    std::string token;
                    std::vector<int> forbidden;
                    for (char c : s + ",") {
                        if (c == ',') {
                            if (!token.empty()) {
                                try { forbidden.push_back(std::stoi(token)); }
                                catch (...) {}
                                token.clear();
                            }
                        } else {
                            token += c;
                        }
                    }
                    for (int lbl : forbidden)
                        neural_policy_->forbidLabel(lbl);
                    if (rank0 && !forbidden.empty()) {
                        std::string msg = "NeuralCprPolicy: forbidden labels:";
                        for (int lbl : forbidden) msg += " " + std::to_string(lbl);
                        msg += " — will select next best allowed label";
                        OpmLog::info(msg);
                    }
                }
            }

            if (rank0) {
                if (neural_policy_->isLoaded())
                    OpmLog::info("NeuralCprPolicy: loaded model from "
                                 + std::string(path)
                                 + " (confidence threshold "
                                 + std::to_string(confidence_threshold_) + ")");
                else
                    OpmLog::info("NeuralCprPolicy: active (rule-based fallback)");
            }

            const char* log_path = std::getenv("OPM_CPR_SWEEP_LOG");
            if (log_path && rank0) {
                sweep_log_ = std::make_unique<std::ofstream>(
                    log_path, std::ios::app);
                if (sweep_log_->is_open() && sweep_log_->tellp() == 0) {
                    *sweep_log_
                        << "config,episode,newton_iter,total_ms,"
                           "dt_days,dt_ratio,nl_residual_norm,nl_residual_reduce,"
                           "nl_iteration,nnz_per_row,diag_dominance,"
                           "prev_linsolver_iters,prev_solve_failed,"
                           "time_elapsed_frac,num_cells_log,block_size,"
                           "well_density,nl_residual_trend,"
                           "num_phases,dt_cut_count,prev2_linsolver_iters,"
                           "condition_number_estimate,bhp_well_fraction\n";
                }
                OpmLog::info("NeuralCprPolicy: sweep logging to "
                             + std::string(log_path));
            }
        }

        /// Format the current prm_ config as "type:weight:fine:coarse".
        std::string currentConfigName() const
        {
            const auto& p = prm_[activeSolverNum_];
            return p.template get<std::string>("preconditioner.type",        "cprw")
                 + ":" + p.template get<std::string>("preconditioner.weight_type",   "trueimpes")
                 + ":" + p.template get<std::string>("preconditioner.finesmoother.type", "dilu")
                 + ":" + p.template get<std::string>(
                             "preconditioner.coarsesolver.preconditioner.smoother", "ilu0");
        }

        /// Write one row to the sweep CSV.
        void writeSweepRow(double total_ms)
        {
            *sweep_log_
                << currentConfigName() << ","
                << last_episode_idx_   << ","
                << int(last_feat_.nl_iteration) << ","
                << total_ms            << ","
                << last_feat_.dt_days              << ","
                << last_feat_.dt_ratio             << ","
                << last_feat_.nl_residual_norm     << ","
                << last_feat_.nl_residual_reduce   << ","
                << last_feat_.nl_iteration         << ","
                << last_feat_.nnz_per_row          << ","
                << last_feat_.diag_dominance       << ","
                << last_feat_.prev_linsolver_iters << ","
                << last_feat_.prev_solve_failed    << ","
                << last_feat_.time_elapsed_frac          << ","
                << last_feat_.num_cells_log              << ","
                << last_feat_.block_size                 << ","
                << last_feat_.well_density               << ","
                << last_feat_.nl_residual_trend          << ","
                << last_feat_.num_phases                 << ","
                << last_feat_.dt_cut_count               << ","
                << last_feat_.prev2_linsolver_iters      << ","
                << last_feat_.condition_number_estimate  << ","
                << last_feat_.bhp_well_fraction          << "\n";
            sweep_log_->flush();
        }

        /// Extract solver-state features from the current matrix and simulator state.
        ///
        /// Caching policy:
        ///   nnz_per_row    — permanent; sparsity pattern never changes.
        ///   diag_dominance — recomputed on Newton iter 0 only; stable across iters.
        CprPolicyFeatures extractFeatures(const Matrix& M)
        {
            CprPolicyFeatures feat;

            const auto& gcomm = simulator_.gridView().comm();

            const double dt_s     = simulator_.timeStepSize();
            feat.dt_days          = dt_s / 86400.0;
            feat.dt_ratio         = (prev_dt_ > 0.0) ? dt_s / prev_dt_ : 1.0;

            // Use the global residual norm so the feature value matches what the
            // sweep (always serial, single-process) recorded during training.
            // In serial gcomm.sum() is a no-op; in parallel it does AllReduce.
            feat.nl_residual_norm = std::sqrt(gcomm.sum(rhs_->two_norm2()));
            feat.nl_residual_reduce = (prev_residual_norm_ > 0.0)
                                     ? feat.nl_residual_norm / prev_residual_norm_
                                     : 1.0;
            feat.nl_iteration = double(
                simulator_.problem().iterationContext().iteration());

            // nnz_per_row: sparsity pattern is fixed for the whole simulation.
            if (cached_nnz_per_row_ < 0.0)
                cached_nnz_per_row_ = (M.N() > 0)
                    ? double(M.nonzeroes()) / double(M.N()) : 7.0;
            feat.nnz_per_row = cached_nnz_per_row_;

            // diag_dominance: recompute only on the first Newton step of each
            // timestep when the matrix reflects the latest linearisation.
            if (feat.nl_iteration == 0.0)
                cached_diag_dominance_ = computeDiagDominance(M);
            feat.diag_dominance = cached_diag_dominance_;

            // Additional features: linear solver history and problem size.
            feat.prev_linsolver_iters = double(cached_linsolver_iters_);
            feat.prev_solve_failed    = last_solve_failed_ ? 1.0 : 0.0;

            const double endT = simulator_.endTime();
            feat.time_elapsed_frac = (endT > 0.0)
                ? std::clamp(simulator_.time() / endT, 0.0, 1.0) : 0.0;

            // Global cell count so num_cells_log matches the serial-sweep value.
            {
                const auto global_N = gcomm.sum(static_cast<std::size_t>(M.N()));
                feat.num_cells_log = (global_N > 0)
                    ? std::log10(static_cast<double>(global_N)) : 1.0;
            }

            feat.block_size = static_cast<double>(Matrix::block_type::rows);

            // well_density: global wells / global cells.
            if (feat.nl_iteration == 0.0) {
                const auto local_wells = static_cast<std::size_t>(
                    simulator_.problem().wellModel().numLocalWells());
                const auto global_wells = gcomm.sum(local_wells);
                const auto global_cells = gcomm.sum(static_cast<std::size_t>(M.N()));
                cached_num_wells_ = static_cast<double>(global_wells);
                cached_global_cells_ = static_cast<double>(global_cells);
            }
            feat.well_density = (cached_global_cells_ > 0.0)
                ? cached_num_wells_ / cached_global_cells_ : 0.0;

            // nl_residual_trend: ratio of the previous two residual norms.
            feat.nl_residual_trend = (prev2_residual_norm_ > 0.0 && prev_residual_norm_ > 0.0)
                ? prev_residual_norm_ / prev2_residual_norm_
                : 1.0;

            // num_phases: count of active fluid phases (1/2/3).
            {
                int np = 0;
                if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) ++np;
                if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx))   ++np;
                if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx))   ++np;
                feat.num_phases = static_cast<double>(np);
            }

            // dt_cut_count: consecutive timestep cuts (detected on Newton iter 0).
            // A cut is signalled by dt_ratio < 0.99 at the start of a new timestep.
            if (feat.nl_iteration == 0.0) {
                if (prev_dt_ > 0.0 && feat.dt_ratio < 0.99)
                    dt_cut_count_++;
                else
                    dt_cut_count_ = 0;
            }
            feat.dt_cut_count = static_cast<double>(dt_cut_count_);

            // prev2_linsolver_iters: Krylov iterations two solves ago (trend signal).
            feat.prev2_linsolver_iters = static_cast<double>(prev2_linsolver_iters_);

            // condition_number_estimate: max/min diagonal block norm over sampled rows.
            if (feat.nl_iteration == 0.0)
                cached_condition_number_ = computeConditionEstimate(M);
            feat.condition_number_estimate = cached_condition_number_;

            // bhp_well_fraction: global fraction of wells on BHP control.
            if (feat.nl_iteration == 0.0) {
                const auto& ws = simulator_.problem().wellModel().wellState();
                const int nw = static_cast<int>(ws.numWells());
                int bhp_count = 0;
                for (int w = 0; w < nw; ++w) {
                    const auto& s = ws.well(w);
                    if (s.production_cmode == WellProducerCMode::BHP ||
                        s.injection_cmode  == WellInjectorCMode::BHP)
                        ++bhp_count;
                }
                const int global_nw  = gcomm.sum(nw);
                const int global_bhp = gcomm.sum(bhp_count);
                cached_bhp_well_fraction_ = (global_nw > 0)
                    ? double(global_bhp) / double(global_nw) : 0.0;
            }
            feat.bhp_well_fraction = cached_bhp_well_fraction_;

            return feat;
        }

        /// Cheap diagonal-dominance proxy: sample up to 200 block rows.
        static double computeDiagDominance(const Matrix& M)
        {
            const std::size_t nSample = std::min(M.N(), std::size_t(200));
            double sumRatio = 0.0;
            std::size_t count = 0;
            for (std::size_t i = 0; i < nSample; ++i) {
                const auto& row = M[i];
                double diagN = 0.0, offN = 0.0;
                for (auto it = row.begin(); it != row.end(); ++it) {
                    const double fn = it->frobenius_norm();
                    if (it.index() == i) diagN += fn;
                    else                  offN  += fn;
                }
                if (offN > 0.0) { sumRatio += diagN / offN; ++count; }
            }
            return (count > 0) ? sumRatio / double(count) : 1.0;
        }

        /// Cheap condition-number proxy: max/min diagonal block norm over sampled rows.
        static double computeConditionEstimate(const Matrix& M)
        {
            const std::size_t nSample = std::min(M.N(), std::size_t(200));
            double maxD = 0.0, minD = std::numeric_limits<double>::max();
            for (std::size_t i = 0; i < nSample; ++i) {
                const auto& row = M[i];
                for (auto it = row.begin(); it != row.end(); ++it) {
                    if (it.index() == i) {
                        const double fn = it->frobenius_norm();
                        maxD = std::max(maxD, fn);
                        minD = std::min(minD, fn);
                    }
                }
            }
            return (minD > 0.0) ? maxD / minD : 1.0;
        }

        /// Run the neural policy and update prm_ for the upcoming solver build.
        ///
        /// Three-tier optimisation:
        ///   Tier 1 — early exit when the solver won't be rebuilt (prm_ is not
        ///            re-read by the preconditioner update() path).
        ///   Tier 2 — skip all tree manipulation when the predicted config is
        ///            identical to the one already in prm_.
        ///   Tier 3 — patch only the 1–3 changed fields when the preconditioner
        ///            type is unchanged, rather than rebuilding all 30+ nodes.
        void applyNeuralPolicy(const Matrix& M)
        {
            // Tier 1: prm_ will not be re-read if we are not about to create a
            // new solver object.  Skip early.
            if (!shouldCreateSolver())
                return;

            CprPolicyFeatures feat = extractFeatures(M);
            float confidence = 0.0f;
            CprPolicyAction cfg = neural_policy_->predict(feat, &confidence);

            // On a failed previous solve revert to the safe default config.
            if (last_solve_failed_)
                cfg = CprPolicyAction{};

            // In a parallel run each MPI rank predicts from its own local matrix
            // partition, so ranks can independently arrive at different configs.
            // Rank 0 is authoritative: encode its decision as a label 0-23, or
            // -1 (below-threshold / keep current config), then broadcast so every
            // rank applies the same preconditioner structure.  Without this, ranks
            // build mismatched solver objects and deadlock in AMG / overlap setup.
            if (isParallel()) {
                // -1 means "keep current config" (below-threshold case).
                int label = -1;
                if (simulator_.gridView().comm().rank() == 0) {
                    if (!policy_cfg_valid_ || confidence >= confidence_threshold_) {
                        const int dec    = static_cast<int>(cfg.weight_type);
                        const int cprw   = cfg.use_cprw ? 1 : 0;
                        const int fine   = (cfg.fine_smoother   == CprSmoother::DILU) ? 1 : 0;
                        const int coarse = (cfg.coarse_smoother == CprSmoother::DILU) ? 1 : 0;
                        label = dec * 8 + cprw * 4 + fine * 2 + coarse;
                    }
                }
                simulator_.gridView().comm().broadcast(&label, 1, 0);
                if (label < 0)
                    return;
                const int d = label / 8, w = (label / 4) % 2,
                          f = (label / 2) % 2, c = label % 2;
                switch (d) {
                case 0:  cfg.weight_type = CprWeightType::QuasiIMPES;         break;
                case 1:  cfg.weight_type = CprWeightType::TrueIMPES;          break;
                default: cfg.weight_type = CprWeightType::TrueIMPESAnalytic;  break;
                }
                cfg.use_cprw        = (w == 1);
                cfg.fine_smoother   = (f == 0) ? CprSmoother::ParOverILU0 : CprSmoother::DILU;
                cfg.coarse_smoother = (c == 0) ? CprSmoother::ILU0        : CprSmoother::DILU;
            } else {
                // Serial path: keep existing threshold check.
                if (policy_cfg_valid_ && confidence < confidence_threshold_)
                    return;
            }

            const bool policy_changed = !policy_cfg_valid_ || !(cfg == last_policy_cfg_);
            if (simulator_.gridView().comm().rank() == 0) {
                OpmLog::debug("NeuralCprPolicy decision label_"
                              + std::to_string(policyLabel(cfg))
                              + " confidence=" + std::to_string(confidence)
                              + " changed=" + std::string(policy_changed ? "yes" : "no")
                              + " " + policyLogStr(cfg));
            }

            // Tier 2: config unchanged — nothing to do.
            if (!policy_changed)
                return;

            const double tol   = prm_[activeSolverNum_].template
                                 get<double>("tol",     0.005);
            const int  maxiter = prm_[activeSolverNum_].template
                                 get<int>("maxiter", 20);

            if (policy_cfg_valid_ && cfg.use_cprw == last_policy_cfg_.use_cprw) {
                // Tier 3: same preconditioner type — patch only the changed keys.
                NeuralCprPolicy::updatePropertyTreeInPlace(
                    cfg, last_policy_cfg_, prm_[activeSolverNum_]);
            } else {
                // Full rebuild needed (first call or type switch).
                // If the preconditioner type changed, the existing FlexibleSolver
                // object cannot be reused — force recreation.
                if (policy_cfg_valid_
                    && !flexibleSolver_.empty()
                    && flexibleSolver_[activeSolverNum_].solver_)
                {
                    flexibleSolver_[activeSolverNum_].solver_.reset();
                }
                prm_[activeSolverNum_] = NeuralCprPolicy::configToPropertyTree(
                                             cfg, tol, maxiter);
            }

            if (simulator_.gridView().comm().rank() == 0)
                OpmLog::debug("NeuralCprPolicy: " + policyLogStr(cfg));

            last_policy_cfg_    = cfg;
            policy_cfg_valid_   = true;
            prev_dt_            = simulator_.timeStepSize();
            prev2_residual_norm_ = prev_residual_norm_;   // shift history
            prev_residual_norm_  = feat.nl_residual_norm;
        }

        static std::string policyLogStr(const CprPolicyAction& a)
        {
            return std::string(a.use_cprw ? "cprw" : "cpr")
                   + " weight=" + NeuralCprPolicy::weightTypeStr(a.weight_type)
                   + " fine="   + NeuralCprPolicy::fineSmootherStr(a.fine_smoother)
                   + " coarse=" + NeuralCprPolicy::coarseSmootherStr(a.coarse_smoother);
        }

        static int policyLabel(const CprPolicyAction& a)
        {
            const int dec = static_cast<int>(a.weight_type);
            const int cprw = a.use_cprw ? 1 : 0;
            const int fine = (a.fine_smoother == CprSmoother::DILU) ? 1 : 0;
            const int coarse = (a.coarse_smoother == CprSmoother::DILU) ? 1 : 0;
            return dec * 8 + cprw * 4 + fine * 2 + coarse;
        }

        void setActiveSolver(const int num) override
        {
            if (num > static_cast<int>(prm_.size()) - 1) {
                OPM_THROW(std::logic_error, "Solver number " + std::to_string(num) + " not available.");
            }
            activeSolverNum_ = num;
            if (simulator_.gridView().comm().rank() == 0) {
                OpmLog::debug("Active solver = " + std::to_string(activeSolverNum_)
                              + " (" + parameters_[activeSolverNum_].linsolver_ + ")");
            }
        }

        int numAvailableSolvers() const override
        {
            return flexibleSolver_.size();
        }

        void initPrepare(const Matrix& M, Vector& b)
        {
            // matrix_ starts out null, so this also covers the first call.
            const bool matrix_changed = &M != matrix_;

            if (matrix_changed) {
                // The matrix object is no longer the one the solver was built
                // for, so the solver has to be rebuilt rather than updated.
                force_recreate_ = true;

                // model will not change the matrix object. Hence simply store a pointer
                // to the original one with a deleter that does nothing.
                // Outch! We need to be able to scale the linear system! Hence const_cast
                matrix_ = const_cast<Matrix*>(&M);

                useWellConn_ = Parameters::Get<Parameters::MatrixAddWellContributions>();
                // setup sparsity pattern for jacobi matrix for preconditioner (only used for openclSolver)
            }
            rhs_ = &b;

            // TODO: check all solvers, not just one.
            // We use lower case as the internal canonical representation of solver names
            std::string type = prm_[activeSolverNum_].template get<std::string>("preconditioner.type", "paroverilu0");
            std::ranges::transform(type, type.begin(), ::tolower);
            if (isParallel() && type != "paroverilu0") {
                detail::makeOverlapRowsInvalid(getMatrix(), overlapRows_);
            }
        }

        void prepare(const SparseMatrixAdapter& M, Vector& b) override
        {
            prepare(M.istlMatrix(), b);
        }

        void prepare(const Matrix& M, Vector& b) override
        {
            OPM_TIMEBLOCK(istlSolverPrepare);
            try {
                initPrepare(M, b);

                if (sweep_log_) {
                    // Sweep mode: record features; use static --linear-solver config.
                    last_feat_        = extractFeatures(M);
                    last_episode_idx_ = simulator_.episodeIndex();
                    // Keep prev_* in sync so dt_ratio and nl_residual_reduce are
                    // meaningful on the next Newton step (these are normally updated
                    // in applyNeuralPolicy() which is skipped in sweep mode).
                    prev_dt_             = simulator_.timeStepSize();
                    prev2_residual_norm_ = prev_residual_norm_;          // shift history
                    prev_residual_norm_  = last_feat_.nl_residual_norm;
                    auto t0 = std::chrono::steady_clock::now();
                    prepareFlexibleSolver();
                    last_setup_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                } else {
                    if (neural_policy_)
                        applyNeuralPolicy(M);
                    prepareFlexibleSolver();
                }
            }
            catch (const Dune::MatrixBlockError&) {
                // Construction or update may have left a partial preconditioner.
                // Rebuild it on the next attempt, independently of state rollback.
                eraseMatrix();
                // A singular matrix block found while building the
                // preconditioner is recoverable: rethrow it unchanged so that
                // the adaptive time stepping can chop the time step instead of
                // aborting the run.
                throw;
            }
            OPM_CATCH_AND_RETHROW_AS_CRITICAL_ERROR("This is likely due to a faulty linear solver JSON specification. Check for errors related to missing nodes.");
        }


        void setResidual(Vector& /* b */) override
        {
            // rhs_ = &b; // Must be handled in prepare() instead.
        }

        void getResidual(Vector& b) const override
        {
            b = *rhs_;
        }

        void setMatrix(const SparseMatrixAdapter& /* M */) override
        {
            // matrix_ = &M.istlMatrix(); // Must be handled in prepare() instead.
        }

        int getSolveCount() const override {
            return solveCount_;
        }

        void resetSolveCount() {
            solveCount_ = 0;
        }

        bool solve(Vector& x) override
        {
            OPM_TIMEBLOCK(istlSolverSolve);
            ++solveCount_;
            // Write linear system if asked for.
            const int verbosity = prm_[activeSolverNum_].get("verbosity", 0);
            const bool write_matrix = verbosity > 10;
            if (write_matrix) {
                Helper::writeSystem(simulator_, //simulator is only used to get names
                                    getMatrix(),
                                    *rhs_,
                                    comm_.get());
            }

            // Solve system.
            Dune::InverseOperatorResult result;
            {
                OPM_TIMEBLOCK(flexibleSolverApply);
                assert(flexibleSolver_[activeSolverNum_].solver_);
                auto t0 = std::chrono::steady_clock::now();
                flexibleSolver_[activeSolverNum_].solver_->apply(x, *rhs_, result);
                if (sweep_log_) {
                    double solve_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    writeSweepRow(last_setup_ms_ + solve_ms);
                }
            }

            iterations_              = result.iterations;
            prev2_linsolver_iters_   = cached_linsolver_iters_;  // shift history
            cached_linsolver_iters_  = result.iterations;

            // Check convergence, iterations etc.
            last_solve_failed_ = !checkConvergence(result);
            return !last_solve_failed_;
        }


        /// Solve the system of linear equations Ax = b, with A being the
        /// combined derivative matrix of the residual and b
        /// being the residual itself.
        /// \param[in] residual   residual object containing A and b.
        /// \return               the solution x

        /// \copydoc NewtonIterationBlackoilInterface::iterations
        int iterations () const override { return iterations_; }

        /// \copydoc NewtonIterationBlackoilInterface::parallelInformation
        const std::any& parallelInformation() const { return parallelInformation_; }

        const CommunicationType* comm() const override { return comm_.get(); }

        void setDomainIndex(const int index)
        {
            domainIndex_ = index;
        }

        bool isNlddLocalSolver() const
        {
            return parameters_[activeSolverNum_].is_nldd_local_solver_;
        }

    protected:
#if HAVE_MPI
        using Comm = Dune::OwnerOverlapCopyCommunication<int, int>;
#endif

        bool checkConvergence(const Dune::InverseOperatorResult& result) const
        {
            return AbstractISTLSolver<SparseMatrixAdapter, Vector>::checkConvergence(result, parameters_[activeSolverNum_]);
        }

        bool isParallel() const {
#if HAVE_MPI
            return !forceSerial_ && comm_->communicator().size() > 1;
#else
            return false;
#endif
        }

        void prepareFlexibleSolver()
        {
            OPM_TIMEBLOCK(flexibleSolverPrepare);
            if (shouldCreateSolver()) {
                if (!useWellConn_) {
                    if (isNlddLocalSolver()) {
                        auto wellOp = std::make_unique<DomainWellModelAsLinearOperator<WellModel, Vector, Vector>>(simulator_.problem().wellModel());
                        wellOp->setDomainIndex(domainIndex_);
                        flexibleSolver_[activeSolverNum_].wellOperator_ = std::move(wellOp);
                    }
                    else {
                        auto wellOp = std::make_unique<WellModelOperator>(simulator_.problem().wellModel());
                        flexibleSolver_[activeSolverNum_].wellOperator_ = std::move(wellOp);
                    }
                }
                std::function<Vector()> weightCalculator
                    = this->getWeightsCalculator(prm_[activeSolverNum_], pressureIndex);
                OPM_TIMEBLOCK(flexibleSolverCreate);
                flexibleSolver_[activeSolverNum_].create(getMatrix(),
                                                         isParallel(),
                                                         prm_[activeSolverNum_],
                                                         pressureIndex,
                                                         weightCalculator,
                                                         forceSerial_,
                                                         comm_.get());
                numWellEquations_ = this->numWellEquations();
            }
            else
            {
                OPM_TIMEBLOCK(flexibleSolverUpdate);
                flexibleSolver_[activeSolverNum_].pre_->update();
            }
        }


        /// Number of extra equations the well operator contributes.
        int numWellEquations() const
        {
            return useWellConn_
                ? 0
                : simulator_.problem().wellModel().numLocalWellsEnd();
        }

        /// Return true if we should (re)create the whole solver,
        /// instead of just calling update() on the preconditioner.
        bool shouldCreateSolver() const
        {
            // Decide if we should recreate the solver or just do
            // a minimal preconditioner update.
            if (force_recreate_) {
                force_recreate_ = false; // one-shot trigger set by initPrepare
                return true;
            }

            if (flexibleSolver_.empty()) {
                return true;
            }

            if (!flexibleSolver_[activeSolverNum_].solver_) {
                return true;
            }

            // The CPRW coarse system reserves a row per well, so its size is
            // fixed when the solver is built.  A well drilled by an ACTIONX
            // block changes the well count mid-run; updating in place would
            // then write past the end of the coarse matrix.
            if (this->numWellEquations() != numWellEquations_) {
                return true;
            }

            if (flexibleSolver_[activeSolverNum_].pre_->hasPerfectUpdate()) {
                return false;
            }

            // For AMG based preconditioners, the hierarchy depends on the matrix values
            // so it is recreated at certain intervals
            if (this->parameters_[activeSolverNum_].cpr_reuse_setup_ == 0) {
                // Always recreate solver.
                return true;
            }
            if (this->parameters_[activeSolverNum_].cpr_reuse_setup_ == 1) {
                // Recreate solver on the first iteration of every timestep.
                return this->simulator_.problem().iterationContext().isFirstGlobalIteration();
            }
            if (this->parameters_[activeSolverNum_].cpr_reuse_setup_ == 2) {
                // Recreate solver if the last solve used more than 10 iterations.
                return this->iterations() > 10;
            }
            if (this->parameters_[activeSolverNum_].cpr_reuse_setup_ == 3) {
                // Never recreate the solver
                return false;
            }
            if (this->parameters_[activeSolverNum_].cpr_reuse_setup_ == 4) {
                // Recreate solver every 'step' solve calls.
                const int step = this->parameters_[activeSolverNum_].cpr_reuse_interval_;
                const bool create = ((solveCount_ % step) == 0);
                return create;
            }
            // If here, we have an invalid parameter.
            const bool on_io_rank = (simulator_.gridView().comm().rank() == 0);
            std::string msg = "Invalid value: " + std::to_string(this->parameters_[activeSolverNum_].cpr_reuse_setup_)
                + " for --cpr-reuse-setup parameter, run with --help to see allowed values.";
            if (on_io_rank) {
                OpmLog::error(msg);
            }
            throw std::runtime_error(msg);

            return false;
        }


        // Weights to make approximate pressure equations.
        // Calculated from the storage terms (only) of the
        // conservation equations, ignoring all other terms.
        std::function<Vector()> getWeightsCalculator(const PropertyTree& prm,
                                                     std::size_t pressIndex) const
        {
            std::function<Vector()> weightsCalculator;

            using namespace std::string_literals;

            auto preconditionerType = prm.get("preconditioner.type"s, "cpr"s);
            // We use lower case as the internal canonical representation of solver names
            std::ranges::transform(preconditionerType, preconditionerType.begin(), ::tolower);
            if (preconditionerType == "cpr" || preconditionerType == "cprt"
                || preconditionerType == "cprw" || preconditionerType == "cprwt") {
                const bool transpose = preconditionerType == "cprt" || preconditionerType == "cprwt";
                const bool enableThreadParallel = this->parameters_[0].cpr_weights_thread_parallel_;
                const auto weightsType = prm.get("preconditioner.weight_type"s, "quasiimpes"s);
                if (weightsType == "quasiimpes") {
                    // Use the current matrix: preconditioner updates call this
                    // without rebuilding the solver.
                    weightsCalculator = [this, transpose, pressIndex, enableThreadParallel]() {
                        return Amg::getQuasiImpesWeights<Matrix, Vector>(
                            getMatrix(), pressIndex, transpose, enableThreadParallel);
                    };
                } else if ( weightsType == "trueimpes" ) {
                    weightsCalculator =
                        [this, pressIndex, enableThreadParallel]
                        {
                            Vector weights(rhs_->size());
                            ElementContext elemCtx(simulator_);
                            Amg::getTrueImpesWeights(pressIndex,
                                                     weights,
                                                     elemCtx,
                                                     simulator_.model(),
                                                     *element_chunks_,
                                                     enableThreadParallel
                            );
                            return weights;
                        };
                } else if  (weightsType == "trueimpesanalytic" ) {
                    weightsCalculator =
                        [this, pressIndex, enableThreadParallel]
                        {
                            Vector weights(rhs_->size());
                            ElementContext elemCtx(simulator_);
                            Amg::getTrueImpesWeightsAnalytic(pressIndex,
                                                             weights,
                                                             elemCtx,
                                                             simulator_.model(),
                                                             *element_chunks_,
                                                             enableThreadParallel
                            );
                            return weights;
                        };
                } else {
                    OPM_THROW(std::invalid_argument,
                              "Weights type " + weightsType +
                              "not implemented for cpr."
                              " Please use quasiimpes, trueimpes or trueimpesanalytic.");
                }
            }
            return weightsCalculator;
        }


        Matrix& getMatrix()
        {
            return *matrix_;
        }

        const Matrix& getMatrix() const
        {
            return *matrix_;
        }

        const Simulator& simulator_;
        mutable int iterations_;
        mutable int solveCount_;
        std::any parallelInformation_;

        // non-const to be able to scale the linear system
        Matrix* matrix_;
        Vector *rhs_;

        int activeSolverNum_ = 0;
        std::vector<detail::FlexibleSolverInfo<Matrix,Vector,CommunicationType>> flexibleSolver_;
        std::vector<int> overlapRows_;
        std::vector<int> interiorRows_;

        int domainIndex_ = -1;

        // Number of well equations the current solver was built for, or -1 if
        // no solver has been built yet.  See shouldCreateSolver().
        int numWellEquations_ = -1;

        bool useWellConn_;

        std::vector<FlowLinearSolverParameters> parameters_;
        bool forceSerial_ = false;
        std::vector<PropertyTree> prm_;

        std::shared_ptr< CommunicationType > comm_;
        std::unique_ptr<ElementChunksType> element_chunks_;
        // set when initPrepare detects a different matrix object; cleared by
        // shouldCreateSolver() (hence mutable in an otherwise const query)
        mutable bool force_recreate_ = false;

        // Neural CPR policy members
        std::unique_ptr<NeuralCprPolicy> neural_policy_;
        float           confidence_threshold_ = 0.6f;  ///< min softmax prob to allow a config switch
        bool            last_solve_failed_    = false;
        bool            policy_cfg_valid_     = false;  ///< true once last_policy_cfg_ is set
        CprPolicyAction last_policy_cfg_;               ///< last action written into prm_
        double          prev_dt_              = 0.0;    ///< dt from the previous prepare() call (s)
        double          prev_residual_norm_   = 0.0;    ///< rhs norm from the previous prepare() call
        double          prev2_residual_norm_  = 0.0;    ///< rhs norm from two prepare() calls ago
        int             cached_linsolver_iters_  = 0;  ///< Krylov iterations from last solve()
        int             prev2_linsolver_iters_   = 0;  ///< Krylov iterations two solves ago
        int             dt_cut_count_            = 0;  ///< consecutive timestep cuts

        // Feature caches (avoid recomputing invariants)
        mutable double cached_nnz_per_row_         = -1.0;  ///< permanent after first call
        mutable double cached_diag_dominance_       =  1.0;  ///< refreshed on Newton iter 0
        mutable double cached_num_wells_            =  0.0;  ///< global well count, refreshed on iter 0
        mutable double cached_global_cells_         =  0.0;  ///< global cell count, refreshed on iter 0
        mutable double cached_condition_number_     =  1.0;  ///< refreshed on Newton iter 0
        mutable double cached_bhp_well_fraction_    =  0.0;  ///< refreshed on Newton iter 0

        // Sweep logging (OPM_CPR_SWEEP_LOG)
        std::unique_ptr<std::ofstream> sweep_log_;
        CprPolicyFeatures last_feat_{};          ///< features from last extractFeatures()
        int               last_episode_idx_ = 0; ///< simulator episode index at last prepare()
        double            last_setup_ms_    = 0.0; ///< prepareFlexibleSolver() wall time (ms)
    }; // end ISTLSolver

} // namespace Opm

#endif // OPM_ISTLSOLVER_HEADER_INCLUDED
