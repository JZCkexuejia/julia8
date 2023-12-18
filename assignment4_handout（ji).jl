using LinearAlgebra

"""
Computes the maximum magnitude eigenpair for the real symmetric matrix A 
with the power method. Ensures that the error tolerance is satisfied by 
using the Bauer-Fike theorem.

Inputs:
    A: real symmetric matrix
    tol: error tolerance; i.e., the maximum eigenvalue estimate 
         must be within tol of the true maximum eigenvalue

Outputs:
    λ: the estimate of the maximum magnitude eigenvalue
    v: the estimate of the normalized eigenvector corresponding to λ
"""
function power_method_symmetric(A, tol)
    λ, v = 0, ones(size(A, 1))

    while true
        v = A * v
        v /= norm(v)
        λ = v' * A * v

        if norm(A * v - λ * v) <= tol
            break
        end
    end

    return λ, v
end

"""
Compute the PageRank algorithm, as described in the assignment handout, 
on a directed graph described by its edges. In the description of inputs 
below, m is the number of directed edges, and n is the number of vertices. 
Assume that the vertices are named 1, 2, ..., n. 

Inputs:
    edges: a m-by-2 matrix of integers containing directed edges, where 
           the first column contains the source of each edge and the 
           second column contains the destination of each edge.  
    tol: error tolerance; i.e., the maximum eigenvalue estimate 
         must be within tol of the true maximum eigenvalue of 1

Outputs:
    v: the eigenvector corresponding to the eigenvalue λ₁ = 1 for the 
       transition matrix induced by the directed graph described by 
       the input edges. 
"""
function page_rank(edges, tol)

    # Get number or verticies
    n = length(unique(edges))
    m = size(edges, 1)

    # Organize the destinations for source nodes
    count = [[] for _ in 1:n]

    for i in 1:m
        push!(count[edges[i, 1]], edges[i, 2])
    end

    # Create and fill transition matrix
    P = zeros(n, n)

    for i in 1:n
        if length(count[i]) == 0
            P[:, i] .= 1 / n # Dead-end webpages randomly land on any webpage with equal probability
        else
            for j in count[i]
                P[j, i] = 1 / length(count[i])
            end
        end
    end

    # Calculate the eigenvector corresponding to eigenvalue λ₁ = 1
    v = ones(size(P, 1))

    while true
        v = P * v
        v /= norm(v, 1)

        if norm(P * v - v, 1) <= tol
            break
        end
    end

    return v
end

# Why does this oscillate? Because eigenvalues are -1 and 1?
# edges = [1 2; 2 1; 2 3; 3 2]
# page_rank(edges, 1e-6)