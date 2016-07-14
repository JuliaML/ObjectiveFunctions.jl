module RegularizerTest

using Base.Test
using ObjectiveFunctions

# Elastic Net
enet = ElasticNetPenalty(1.0,1.0)
x = ones(8)
@test isapprox(value(enet,x),12)
@test isapprox(value(ElasticNetPenalty(),x),12)

# L1 Penalty
l1 = L1Penalty()
@test isapprox(value(l1,x),8)

# L2 Penalty
l2 = L2Penalty()
@test isapprox(value(l2,x),4)

end # module

using RegularizerTest
