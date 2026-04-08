using Cbc, JuMP, SparseArrays, DelimitedFiles

K1 = [
300 140 40
]
K2 = [
500 230 60
]
K3 = [
1000 400 70
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

function solveIP(H)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    # myModel = Model(GLPK.Optimizer)

    A1 = constructA(H,K1)
    A2 = constructA(H,K2)
    A3 = constructA(H,K3)

    @variable(myModel, x1[1:h], Bin )
    @variable(myModel, x2[1:h], Bin )
    @variable(myModel, x3[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    # @variable(myModel, theta)

    # @objective(myModel, Min, sum(x[j] for j=1:h) )
    # @objective(myModel, Min, sum(abs(R[j]-H[j]-10) for j=1:h) )
    @objective(myModel, Min, sum(R[j] - H[j] - 10  for j=1:h))
    # @constraint(myModel, [j=1:h],theta >= R[j] - H[j] - 10 )
    # @constraint(myModel, [j=1:h],theta >= -(R[j] - H[j] - 10) )


    @constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h],R[i] == sum(A1[i,j]*x1[j] + A2[i,j]*x2[j] + A3[i,j]*x3[j] for j=1:h) )

    # only one bomb per location and no adjacent bombs
    @constraint(myModel, [j=1:h-1], x1[j] + x1[j+1] + x2[j] + x2[j+1] + x3[j] + x3[j+1] <= 1)


    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        x1_val = JuMP.value.(x1)
        x2_val = JuMP.value.(x2)
        x3_val = JuMP.value.(x3)
        R_val = JuMP.value.(R)
        println("x1 = ", x1_val)
        println("x2 = ", x2_val)
        println("x3 = ", x3_val)
        println("R = ", R_val)

        # Save H, x, R to a TSV file for Python
        # using DelimitedFiles
        return x1_val, x2_val, x3_val, R_val
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

# Read the interpolated heights from the file
data = readdlm("./linear_programming/data/interpolated_heights.txt", '\t', skipstart=1)
data = data[1:250, :]
H = data[:, 2]  # Second column is the interpolated heights

x1_val, x2_val, x3_val, R_val = solveIP(H)
distance = data[:, 1]
output = [distance x1_val x2_val x3_val R_val]
header = "distance\tx1\tx2\tx3\tR"
open("./linear_programming/data/smooth_no_adj_dial_solution_results.txt", "w") do io
    write(io, header * "\n")
    writedlm(io, output, '\t')
end

# for all 313 entries:
#Cbc0010I After 76000 nodes, 14318 on tree, 74280 best solution, best possible 72411.005 (1112.18 seconds)

# for 250 entries:
# Cbc0001I Search completed - best objective 57120, took 432435 iterations and 6915 nodes (74.06 seconds)

# for 313 entries, but only k1 and k2: