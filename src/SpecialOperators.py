from sympy import sin, cos, acos, atan2, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative, Integral, integrate
from sympy.vector import Vector, cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral, matrix_to_vector
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt


#Initial coordinate systems setup
C = CoordSys3D('C') #(x, y, z)
P = C.create_new('P', transformation='cylindrical', variable_names = ["r", "phi", "z"]) #r, phi, z: (r*cos(phi), r*sin(phi), z)
S = C.create_new('S', transformation='spherical') #r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta))

from ElectroMagneticField.src.Vectors import Cartesian_to_Spherical, Cartesian_to_Cylindrical

def change_derivatives_to_constant_by_denominator(expr, denominator, constant):
    new_expr = expr
    for Derivatives in expr.atoms(Derivative):
        if (Derivatives.variables[0] == denominator):
            new_expr = new_expr.subs(Derivatives, constant)
    return new_expr

def XYZ_to_Function(x_mesh, y_mesh, z_mesh, func_system, func, **kwargs):
    if func_system == C:
        return func(x_mesh, y_mesh, z_mesh, **kwargs)
    elif func_system == P:
        r , p, z = Cartesian_to_Cylindrical(x_mesh, y_mesh, z_mesh)
        polarVector = r*P.i + t*P.j + z*P.k
        ResultMatrix = func(polarVector, **kwargs).to_matrix(P)
        P_to_C_matrix = Matrix([[cos(p), -sin(p), 0], [sin(p), cos(p), 0], [0, 0, 1]])
        return matrix_to_vector(P_to_C_matrix*ResultMatrix)
    elif func_system == S:
        r, t, p = Cartesian_to_Spherical(x_mesh, y_mesh, z_mesh)
        sphericalVector = r*S.i + p*S.j + t*S.k
        ResultMatrix = func(sphericalVector, **kwargs).to_matrix(S)
        S_to_C_matrix = Matrix([
            [sin(t)*cos(p), cos(t)*cos(p), -sin(p)],
            [sin(t)*sin(p), cos(t)*sin(p), cos(p)],
            [cos(t), -sin(t), 0]])
        return matrix_to_vector(S_to_C_matrix*ResultMatrix)
    else:
        raise ValueError("func_system must be a valid coordinate system: C, P or S.")
    





