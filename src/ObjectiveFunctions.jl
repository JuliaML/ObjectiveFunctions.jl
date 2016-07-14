module ObjectiveFunctions

using Reexport
importall LearnBase
@reexport using LearnBase
@reexport using Losses 

# TODO: move losses here.
const DEFAULT_TOL = 1e-8

export  SimplexConstraint,
        AffineConstraint,
        BoxConstraint,
        L2NormEqConstraint,

        LinearLeastSquares,
        LinearLeastSquaresProx,

        NoPenalty,
        L1Penalty,
        L2Penalty,
        ElasticNetPenalty,
        NuclearNormPenalty,
        FirstDerivL2Penalty,

        # These are defined, but not exported in LearnBase.
        # Should LearnBase export everything?
        prox,
        prox!

include("common.jl")
include("penalties.jl")

include("constraints/simplex.jl")
include("constraints/box.jl")
include("constraints/unit_length.jl")
include("constraints/affine.jl")

include("models/linear.jl")

include("regularizers/l1.jl")
include("regularizers/l2.jl")
include("regularizers/elastic_net.jl")
include("regularizers/nuclear_norm.jl")
include("regularizers/smooth.jl")

end # module
