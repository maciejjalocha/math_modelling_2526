using Cbc, JuMP, SparseArrays, DelimitedFiles

K = [
300 140 40
]


function constructA(H,K)
    # Make a function that returns A when given H and K
    h = length(H)
    A = spzeros(h,h)
    for i in 1:h
        A[i,i] = K[1]
    end
    for i in 1:h-1
        A[i,i+1] = K[2]
        A[i+1,i] = K[2]
    end
    for i in 1:h-2
        A[i,i+2] = K[3]
        A[i+2,i] = K[3]
    end
    return A
end

function solveIP(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    # myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    # @variable(myModel, theta[1:h])

    # @objective(myModel, Min, sum(x[j] for j=1:h) )
    # @objective(myModel, Min, sum(abs(R[j]-H[j]-10) for j=1:h) )
    @objective(myModel, Min, sum(R[j] - H[j] - 10  for j=1:h))
    # @constraint(myModel, [j=1:h],theta[j] >= R[j] - H[j] - 10 )
    # @constraint(myModel, [j=1:h],theta[j] >= -(R[j] - H[j] - 10) )


    @constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        x_val = JuMP.value.(x)
        R_val = JuMP.value.(R)
        println("x = ", x_val)
        println("R = ", R_val)

        # Save H, x, R to a TSV file for Python
        # using DelimitedFiles
        return x_val, R_val
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

# Read the interpolated heights from the file
data = readdlm("./linear_programming/data/interpolated_heights.txt", '\t', skipstart=1)
H = data[:, 2]  # Second column is the interpolated heights

x_val, R_val = solveIP(H,K)
distance = data[:, 1]
output = [distance x_val R_val]
header = "distance\tx\tR"
open("./linear_programming/data/smooth_solution_results.txt", "w") do io
    write(io, "distance\tx\tR\n")
    writedlm(io, output, '\t')
end
