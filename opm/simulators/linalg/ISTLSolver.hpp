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
#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/EclipseState/Phase.hpp>
#include <opm/input/eclipse/EclipseState/Runspec.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>
#include <opm/simulators/linalg/CprPolicyFeatures.hpp>
#include <opm/simulators/linalg/NeuralAbortPolicy.hpp>
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
            initAbortPolicy();
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
            // Explicit baseline mode: retain the upstream property tree and
            // bypass both trained predictions and the rule-based fallback.
            const char* disabled = std::getenv("OPM_NEURAL_CPR_DISABLE");
            if (disabled && std::string(disabled) == "1") {
                neural_policy_.reset();
                if (simulator_.gridView().comm().rank() == 0)
                    OpmLog::info("NeuralCprPolicy: disabled (upstream solver defaults)");
                return;
            }
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

            // Guards: rule out configs that cannot work or are redundant on this
            // deck, and set the anchor (the flow-default solver):
            //   no oil phase  -> trueimpesanalytic (labels 16-23) aborts
            //   wells present -> only cprw (cpr ignores the well coupling)
            //   no wells      -> cprw == cpr, so use the cpr labels (anchor 8)
            // OPM_NEURAL_CPR_GUARDS=0 disables the guards.
            const bool has_wells = simulator_.vanguard().schedule().numWells() > 0;
            anchor_label_ = has_wells ? 12 : 8;
            {
                const char* guards_env = std::getenv("OPM_NEURAL_CPR_GUARDS");
                if (!(guards_env && std::string(guards_env) == "0")) {
                    // From the deck: the fluid system is not initialised yet
                    // when the linear solver is constructed.
                    const bool has_oil = simulator_.vanguard().eclState()
                        .runspec().phases().active(Phase::OIL);
                    for (int lbl = 0; lbl < 24; ++lbl) {
                        const bool cprw = (lbl / 4) % 2 == 1;
                        if ((!has_oil && lbl >= 16) || cprw != has_wells)
                            neural_policy_->forbidLabel(lbl);
                    }
                    if (rank0)
                        OpmLog::info(std::string("NeuralCprPolicy: guards for a deck ")
                                     + (has_wells ? "with" : "without") + " wells and "
                                     + (has_oil ? "with" : "without") + " oil; anchor label "
                                     + std::to_string(anchor_label_));
                }
            }

            // OPM_NEURAL_CPR_MIN_GAIN=g selects anchored decisions for cost
            // models (train_cpr_cost.py, exported with tau=1): stay on the anchor
            // unless another config is predicted cheaper by more than g in
            // log-cost (0.05 ≈ 5 %).
            if (const char* gain_env = std::getenv("OPM_NEURAL_CPR_MIN_GAIN")) {
                anchored_mode_ = true;
                min_gain_ = std::stof(gain_env);
                if (rank0)
                    OpmLog::info("NeuralCprPolicy: anchored cost-model decisions, min gain "
                                 + std::to_string(min_gain_));
            }

            // OPM_NEURAL_CPR_ACTIONS=amg: the model (train_cpr_amg_policy.py)
            // chooses the pressure AMG's prolongation damping and coarsening
            // target at each setup rebuild, anchored on the defaults; the CPR
            // variant itself stays as configured.
            if (const char* actions = std::getenv("OPM_NEURAL_CPR_ACTIONS");
                actions && std::string(actions) == "amg") {
                amg_mode_ = true;
                if (rank0)
                    OpmLog::info("NeuralCprPolicy: choosing AMG prolongation damping and "
                                 "coarsening target, min gain " + std::to_string(min_gain_));
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
            // Sweep mode must be decided identically on every rank: prepare()
            // branches on it, and both branches contain collectives
            // (extractFeatures / applyNeuralPolicy).  Only rank 0 owns the file.
            sweep_mode_ = (log_path != nullptr);
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
                           "setup_rebuilt,setup_ms,solve_ms\n";
                           // "well_density,nl_residual_trend,"
                           // "num_phases,dt_cut_count,prev2_linsolver_iters,"
                           // "condition_number_estimate,bhp_well_fraction\n";
                }
                OpmLog::info("NeuralCprPolicy: sweep logging to "
                             + std::string(log_path));
            }
        }

        /// OPM_CPR_ABORT_MODEL=path enables the neural early-abort policy, and
        /// OPM_CPR_ABORT_THRESHOLD (default 0.95) is the probability of the
        /// attempt being chopped above which the solve's iteration budget is set
        /// to zero.  Independent of the config policy; off in sweep mode, which
        /// records unperturbed runs.
        void initAbortPolicy()
        {
            const char* path = std::getenv("OPM_CPR_ABORT_MODEL");
            if (!path || sweep_mode_)
                return;
            const char* thr = std::getenv("OPM_CPR_ABORT_THRESHOLD");
            abort_policy_ = std::make_unique<NeuralAbortPolicy>(path, thr ? std::stod(thr) : 0.95);
            if (!abort_policy_->isLoaded()) {
                abort_policy_.reset();
                return;
            }
            if (simulator_.gridView().comm().rank() == 0)
                OpmLog::info("NeuralAbortPolicy: loaded model from " + std::string(path)
                             + " (abort above probability "
                             + std::to_string(abort_policy_->threshold()) + ")");
        }

        /// Features of the abort policy for the solve about to start, in the
        /// order of train_abort_policy.py (abort_feasibility.py --features
        /// solver): what the linear solver knows about the timestep attempt.
        /// Updates the per-attempt history, so call it once per solve, on all
        /// ranks (the residual norm is a global reduction).
        NeuralAbortPolicy::Features abortFeatures(const bool rebuild)
        {
            constexpr double itsScale = 20.0;  // feature scaling used in training
            constexpr double ageScale = 30.0;
            auto& s = abort_state_;
            const int j = simulator_.problem().iterationContext().iteration();
            if (j == 0) {
                // A new attempt; it starts at the same time as the previous one
                // exactly when that one was chopped.
                const double t = simulator_.time();
                const double dt = simulator_.timeStepSize();
                if (s.have_prev) {
                    const bool chopped = (t == s.attempt_time);
                    s.after_chop = chopped;
                    s.streak = chopped ? s.streak + 1 : 0;
                    s.prev_iters = chopped ? 0.0 : s.attempt_solves / itsScale;
                    s.dt_change = (s.attempt_dt > 0.0 && dt > 0.0)
                        ? std::log10(dt / s.attempt_dt) : 0.0;
                }
                s.have_prev = true;
                s.attempt_time = t;
                s.attempt_dt = dt;
                s.attempt_solves = 0;
                s.its.clear();
                s.res.clear();
            }
            ++s.attempt_solves;
            s.age = rebuild ? 0 : s.age + 1;
            if (j > 0)
                s.its.push_back(cached_linsolver_iters_);
            const double norm = std::sqrt(simulator_.gridView().comm().sum(rhs_->two_norm2()));
            s.res.push_back(std::log10(std::max(norm, 1e-30)));

            const double r = s.res.back();
            int ups = 0;
            for (std::size_t i = 1; i < s.res.size(); ++i)
                ups += s.res[i] > s.res[i - 1];
            const double maxIts = s.its.empty() ? 0.0 : *std::ranges::max_element(s.its);
            return {{
                static_cast<float>(j / itsScale),
                static_cast<float>(s.after_chop),
                static_cast<float>(std::min(s.streak, 5) / 5.0),
                static_cast<float>(s.dt_change),
                static_cast<float>(s.prev_iters),
                static_cast<float>(j > 0 ? cached_linsolver_iters_ / itsScale : 0.0),
                static_cast<float>(maxIts / itsScale),
                static_cast<float>(std::min(s.age, 30) / ageScale),
                static_cast<float>(rebuild),
                static_cast<float>(r),
                static_cast<float>(r - s.res.front()),
                static_cast<float>(s.res.size() > 1 ? r - s.res[s.res.size() - 2] : 0.0),
                static_cast<float>(r - *std::ranges::min_element(s.res)),
                static_cast<float>(double(ups) / s.res.size()),
            }};
        }

        /// Ask the abort policy whether to give up on the current timestep
        /// attempt before this solve.  Rank 0 decides and all ranks follow.
        bool decideAbort()
        {
            // Whether this solve would rebuild the CPR setup, queried without
            // consuming the one-shot recreate trigger.
            const bool force_recreate = force_recreate_;
            const bool rebuild = shouldCreateSolver();
            force_recreate_ = force_recreate;
            const auto x = abortFeatures(rebuild);

            // Safety guards, independent of the model.  Aborting causes a chop,
            // and chops make the next attempt's history look doomed, so an
            // unchecked policy can chop itself into flow's cut limit (seen on
            // an unseen deck).  Hence no abort before an attempt's first solve
            // (no evidence from the attempt yet), and none once three attempts
            // in a row have been chopped (flow's own logic takes over).
            const bool allowed = simulator_.problem().iterationContext().iteration() > 0
                                 && abort_state_.streak < 3;

            const bool rank0 = (simulator_.gridView().comm().rank() == 0);
            int abort = 0;
            double p = 0.0;
            if (rank0 && allowed) {
                p = abort_policy_->doomProbability(x);
                abort = p > abort_policy_->threshold();
            }
            if (isParallel())
                simulator_.gridView().comm().broadcast(&abort, 1, 0);
            if (abort && rank0)
                OpmLog::debug("NeuralAbortPolicy: giving up on the timestep attempt before "
                              "the solve of Newton iteration "
                              + std::to_string(simulator_.problem().iterationContext().iteration())
                              + " (p=" + std::to_string(p) + ")");
            return abort != 0;
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

        /// Write one row to the sweep CSV.  total_ms = setup_ms + solve_ms,
        /// where setup is a full rebuild (setup_rebuilt=1) or only the cheap
        /// preconditioner update() when the CPR setup is reused.
        void writeSweepRow(double solve_ms)
        {
            *sweep_log_
                << currentConfigName() << ","
                << last_episode_idx_   << ","
                << int(last_feat_.nl_iteration) << ","
                << last_setup_ms_ + solve_ms << ","
                << last_feat_.dt_days              << ","
                << last_feat_.dt_ratio             << ","
                << last_feat_.nl_residual_norm     << ","
                << last_feat_.nl_residual_reduce   << ","
                << last_feat_.nl_iteration         << ","
                << last_feat_.nnz_per_row          << ","
                << last_feat_.diag_dominance       << ","
                << last_feat_.prev_linsolver_iters << ","
                << last_feat_.prev_solve_failed    << ","
                << last_feat_.time_elapsed_frac    << ","
                << last_feat_.num_cells_log        << ","
                << last_feat_.block_size           << ","
                << (last_setup_rebuilt_ ? 1 : 0)   << ","
                << last_setup_ms_                  << ","
                << solve_ms                        << "\n";
                // << last_feat_.well_density               << ","
                // << last_feat_.nl_residual_trend          << ","
                // << last_feat_.num_phases                 << ","
                // << last_feat_.dt_cut_count               << ","
                // << last_feat_.prev2_linsolver_iters      << ","
                // << last_feat_.condition_number_estimate  << ","
                // << last_feat_.bhp_well_fraction          << "\n";
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

            // endTime() and startTime() are absolute (POSIX) times; time() is
            // seconds since the start.
            const double duration = simulator_.endTime() - simulator_.startTime();
            feat.time_elapsed_frac = (duration > 0.0)
                ? std::clamp(simulator_.time() / duration, 0.0, 1.0) : 0.0;

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
        void applyNeuralPolicy(const CprPolicyFeatures& feat)
        {
            // Tier 1: prm_ will not be re-read if we are not about to create a
            // new solver object.  Skip early.  shouldCreateSolver() consumes the
            // one-shot force_recreate_ trigger, which prepareFlexibleSolver()
            // must still see, so restore it.
            const bool force_recreate = force_recreate_;
            const bool create = shouldCreateSolver();
            force_recreate_ = force_recreate;
            if (!create)
                return;

            const bool rank0 = (simulator_.gridView().comm().rank() == 0);

            if (amg_mode_) {
                // Choose the AMG parameter levels for the setup about to be
                // built; the default levels after a failed solve.  Rank 0
                // decides, all ranks apply the same.
                int choice[2] = {0, 0};
                float gain = 0.0f;
                if (rank0 && !last_solve_failed_) {
                    const auto [d, t] = neural_policy_->chooseAmgAnchored(feat, min_gain_, &gain);
                    choice[0] = d;
                    choice[1] = t;
                }
                if (isParallel())
                    simulator_.gridView().comm().broadcast(choice, 2, 0);
                const std::string amg = "preconditioner.coarsesolver.preconditioner.";
                auto& prm = prm_[activeSolverNum_];
                prm.put(amg + "prolongationdamping", NeuralCprPolicy::kAmgDamping[choice[0]]);
                prm.put(amg + "coarsenTarget", NeuralCprPolicy::kAmgCoarsenTarget[choice[1]]);
                if (rank0)
                    OpmLog::debug("NeuralCprPolicy AMG decision damping="
                                  + std::to_string(NeuralCprPolicy::kAmgDamping[choice[0]])
                                  + " coarsenTarget="
                                  + std::to_string(NeuralCprPolicy::kAmgCoarsenTarget[choice[1]])
                                  + " gain=" + std::to_string(gain));
                return;
            }

            // Softmax confidence (classifier) or predicted log-cost gain over
            // the anchor (anchored cost-model mode); used for logging.
            float score = 0.0f;
            CprPolicyAction cfg;
            if (anchored_mode_) {
                // Stay on the anchor (the default solver) unless another config
                // is predicted cheaper by more than min_gain_, and return to it
                // after a failed solve.  Each decision is made afresh, so a poor
                // choice does not persist.  Rank 0 decides and broadcasts, since
                // ranks predicting from their own partitions could disagree.
                int label = anchor_label_;
                if (rank0 && !last_solve_failed_)
                    label = neural_policy_->chooseAnchored(feat, anchor_label_,
                                                           min_gain_, &score);
                if (isParallel())
                    simulator_.gridView().comm().broadcast(&label, 1, 0);
                cfg = NeuralCprPolicy::actionFromLabel(label);
            } else {
                cfg = neural_policy_->predict(feat, &score);

                // On a failed previous solve revert to the default solver.
                if (last_solve_failed_)
                    cfg = NeuralCprPolicy::actionFromLabel(anchor_label_);

                // In a parallel run each MPI rank predicts from its own local
                // matrix partition, so ranks can independently arrive at
                // different configs.  Rank 0 is authoritative: encode its
                // decision as a label 0-23, or -1 (below-threshold / keep current
                // config), then broadcast so every rank applies the same
                // preconditioner structure.  Without this, ranks build mismatched
                // solver objects and deadlock in AMG / overlap setup.
                if (isParallel()) {
                    int label = -1;
                    if (rank0 && (!policy_cfg_valid_ || score >= confidence_threshold_))
                        label = policyLabel(cfg);
                    simulator_.gridView().comm().broadcast(&label, 1, 0);
                    if (label < 0)
                        return;
                    cfg = NeuralCprPolicy::actionFromLabel(label);
                } else {
                    // Serial path: keep existing threshold check.
                    if (policy_cfg_valid_ && score < confidence_threshold_)
                        return;
                }
            }

            const bool policy_changed = !policy_cfg_valid_ || !(cfg == last_policy_cfg_);
            if (rank0) {
                OpmLog::debug("NeuralCprPolicy decision label_"
                              + std::to_string(policyLabel(cfg))
                              + (anchored_mode_ ? " gain=" : " confidence=")
                              + std::to_string(score)
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

                if (sweep_mode_) {
                    // Sweep mode: record features; use static --linear-solver config.
                    last_feat_        = extractFeatures(M);
                    last_episode_idx_ = simulator_.episodeIndex();
                    // Keep prev_* in sync so dt_ratio and nl_residual_reduce are
                    // meaningful on the next Newton step (done identically in
                    // policy mode below).
                    prev_dt_             = simulator_.timeStepSize();
                    prev2_residual_norm_ = prev_residual_norm_;          // shift history
                    prev_residual_norm_  = last_feat_.nl_residual_norm;
                    auto t0 = std::chrono::steady_clock::now();
                    prepareFlexibleSolver();
                    last_setup_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                } else {
                    if (neural_policy_) {
                        // Extract features and keep prev_* in sync on every
                        // solve, exactly as in sweep mode, so the policy sees
                        // features computed the way its training data was.
                        const CprPolicyFeatures feat = extractFeatures(M);
                        applyNeuralPolicy(feat);
                        prev_dt_             = simulator_.timeStepSize();
                        prev2_residual_norm_ = prev_residual_norm_;   // shift history
                        prev_residual_norm_  = feat.nl_residual_norm;
                    }
                    // Giving up skips the setup too: solve() fails at once.
                    abort_now_ = abort_policy_ && decideAbort();
                    if (!abort_now_)
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
            if (abort_now_) {
                // The abort policy gave this solve an iteration budget of zero.
                // Fail it the way a solve that hits maxiter fails, so flow chops
                // the timestep now instead of after more doomed iterations.
                abort_now_ = false;
                iterations_ = 0;
                last_solve_failed_ = true;
                OPM_THROW_NOLOG(NumericalProblem,
                                "Convergence failure for linear solver "
                                "(neural abort policy gave up on the timestep attempt).");
            }
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
                    writeSweepRow(solve_ms);
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
                last_setup_rebuilt_ = true;
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
                last_setup_rebuilt_ = false;
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
        bool            anchored_mode_        = false; ///< cost-model decisions (OPM_NEURAL_CPR_MIN_GAIN)
        float           min_gain_             = 0.05f; ///< log-cost gain needed to leave the anchor
        int             anchor_label_         = 12;    ///< flow-default solver (8 on decks without wells)
        bool            amg_mode_             = false; ///< choose AMG parameters (OPM_NEURAL_CPR_ACTIONS=amg)
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
        std::unique_ptr<std::ofstream> sweep_log_;  // rank 0 only
        bool sweep_mode_ = false;                   // set on all ranks
        CprPolicyFeatures last_feat_{};          ///< features from last extractFeatures()
        int               last_episode_idx_ = 0; ///< simulator episode index at last prepare()
        double            last_setup_ms_    = 0.0; ///< prepareFlexibleSolver() wall time (ms)
        bool              last_setup_rebuilt_ = false; ///< last prepareFlexibleSolver() rebuilt the solver

        // Neural early-abort policy (OPM_CPR_ABORT_MODEL)
        std::unique_ptr<NeuralAbortPolicy> abort_policy_;
        bool abort_now_ = false;  ///< prepare() decided to give up on this attempt
        struct AbortState {
            bool   have_prev = false;     ///< an earlier attempt exists
            bool   after_chop = false;    ///< the previous attempt was chopped
            int    streak = 0;            ///< consecutive chopped attempts before this one
            int    attempt_solves = 0;    ///< solves so far in this attempt
            int    age = 0;               ///< solves since the CPR setup was rebuilt
            double attempt_time = 0.0;    ///< start time of this attempt (s)
            double attempt_dt = 0.0;      ///< step size of this attempt (s)
            double dt_change = 0.0;       ///< log10 of step size over the previous attempt's
            double prev_iters = 0.0;      ///< Newton iterations of the previous attempt / 20
            std::vector<double> its;      ///< linear iterations of this attempt's solves
            std::vector<double> res;      ///< log10 residual norms of this attempt's solves
        } abort_state_;
    }; // end ISTLSolver

} // namespace Opm

#endif // OPM_ISTLSOLVER_HEADER_INCLUDED
