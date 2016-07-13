module ConstraintTest

using Base.Test
using ObjectiveFunctions

# Simplex
simplex = SimplexConstraint(1e-3)

@test value(simplex,[1.,0.,0.,0.,0.]) == 0.0
@test value(simplex,[1.,1.,-1.,0.,0.]) == Inf
@test value(simplex,[1.,1.,0.,0.,0.]) == Inf
@test value(simplex,[1.,0.5,0.,0.,0.]) == Inf

# Affine constraint, A*x = b
A = randn(1,3)
b = randn(1)
affine = AffineConstraint(A,b)

x = A \ b
@test value(affine,x) == 0.0
@test value(affine,x+([-0.1,0.1,-0.1])) == Inf

# Box constraint, lb <= x <= ub
n = 10
lb,ub = -rand(n),rand(n)
box = BoxConstraint(lb,ub)

@test value(box,lb) == 0.0
@test value(box,ub) == 0.0
@test value(box,zeros(n)) == 0.0
for _ = 1:10
    t = rand()
    @test value(box,t*lb+(1-t)*ub) == 0.0
end

x = copy(ub); x[1] += 0.1
@test value(box,x) == Inf
x = copy(lb); x[1] -= 0.1
@test value(box,x) == Inf

# Norm Equality constraint
unitlen = L2NormEqConstraint()
x = ones(16);
@test value(unitlen,x) == Inf
@test isapprox(prox(x,unitlen),fill(0.25,16))
@test value(unitlen,fill(0.25,16)) == 0.0

end # module

using ConstraintTest
