# From https://github.com/AntoninGranados/Minidisk

import numpy as np
from copy import deepcopy

def ellipse_through(points):
	if len(points) == 0:
		return np.zeros((2,2))
	
	if len(points) == 1:
		return np.zeros((2,2))
	
	if len(points) == 2:
		p1 = np.array(points[0])
		p2 = np.array(points[1])
		A = np.outer(p1, p1) + np.outer(p2, p2)
		return np.linalg.inv(A)

	if len(points) == 3:
		A = np.array([
			[points[0][0]**2, 2*points[0][0]*points[0][1], points[0][1]**2],
			[points[1][0]**2, 2*points[1][0]*points[1][1], points[1][1]**2],
			[points[2][0]**2, 2*points[2][0]*points[2][1], points[2][1]**2]
		])
		B = np.ones(3)
		try:
			a, b, c = np.linalg.solve(A, B)
		except:
			return np.zeros((2,2))
		return np.array([[a, b],
						[b, c]])
	
	return np.zeros((2,2))

def inside_ellipse(p, ellipse_mat) -> bool:
	return np.inner(p @ ellipse_mat, p) <= 1 + 1e-5

def miniellipse(P, R=[]):
	P_ = deepcopy(P)
	R_ = deepcopy(R)
	if len(P_) == 0 or len(R_) == 3:
		S = ellipse_through(R_)
	else:
		p = P_.pop()
		S = miniellipse(P_, R_)
		if not np.any(S) or not inside_ellipse(p, S):
			S = miniellipse(P_, R_ + [p])
	return S
