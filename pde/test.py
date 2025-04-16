import torch

# Example 3D tensor with shape [A, D, C]
A, D, C = 5, 4, 6
U_face_all = torch.zeros(A, D, C)

# Suppose these are your indices (both of length N)
# They point to rows in dimension 0 and corresponding positions in dimension 2.
N = 3
where_all_0 = torch.tensor([0, 2, 4])  # for the first dimension
where_all_1 = torch.tensor([1, 3, 5])  # for the third dimension

# Let's create neum_vals with shape [N, D]
neum_vals = torch.randn(N, D)

# Now, assign neum_vals into U_face_all at the specified indices.
U_face_all[where_all_0, :, where_all_1] = neum_vals

# U_face_all now holds the new values in the selected slices
print(U_face_all)
