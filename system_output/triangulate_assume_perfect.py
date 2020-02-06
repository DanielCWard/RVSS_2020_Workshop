import numpy as np
import json

def rot_mat(theta):
    rot_mat = np.zeros([2,2])
    rot_mat[0][0] = np.cos(theta)
    rot_mat[0][1] = -np.sin(theta)
    rot_mat[1][0] = np.sin(theta)
    rot_mat[1][1] = np.cos(theta)

    return rot_mat
def alpha_vec(alpha):
    B = np.zeros([1,2])
    B[0][0] = -np.sin(alpha)
    B[0][1] = np.cos(alpha)

    return B


image_txt_path = 'images.json'
with open(image_txt_path, 'r') as f:
    image_txt = json.load(f)

n=10
robo_x = np.random.rand(10,1)
robo_y = np.random.rand(10,1)

theta = np.radians(np.random.rand(10))

alpha = np.radians(np.random.rand(10))

B_stack = np.array( np.zeros([1]))
A_stack = np.array( np.zeros([1,2]) )

for i in range(n):
    coord_alpha = np.asmatrix(alpha_vec(alpha[i]))
    rot_theta_T =  np.asmatrix( np.transpose(rot_mat( theta[i] )))
    robo_coord = np.asmatrix(np.array([robo_x[i], robo_y[i]]).reshape(2,1))

    B = np.matmul(rot_theta_T, robo_coord)
    B = np.matmul(coord_alpha, B)

    B_stack = np.vstack( (B_stack, B) )

    A = np.matmul( coord_alpha, rot_theta_T )

    A_stack = np.vstack( (A_stack, A) )

A_stack = A_stack[1:,:]
B_stack = B_stack[1:,:]

L = np.linalg.inv( A_stack.transpose() * A_stack ) * A_stack.transpose() * B_stack


