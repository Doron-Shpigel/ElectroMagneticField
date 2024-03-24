from sympy import sin, cos, acos, atan2, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative, Integral, integrate, diff
from sympy.vector import Vector, cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral, matrix_to_vector, is_conservative
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt


#Initial coordinate systems setup
C = CoordSys3D('C') #(x, y, z)
P = C.create_new('P', transformation='cylindrical', variable_names = ["r", "phi", "z"]) #r, phi, z: (r*cos(phi), r*sin(phi), z)
S = C.create_new('S', transformation='spherical') #r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta))


def change_derivatives_to_constant_by_denominator(expr, denominator, constant):
    new_expr = expr
    for Derivatives in expr.atoms(Derivative):
        if (Derivatives.variables[0] == denominator):
            new_expr = new_expr.subs(Derivatives, constant)
    return new_expr

def XYZ_to_Function(x_mesh, y_mesh, z_mesh, func_system, func, **kwargs):
    from .Vectors import Cartesian_to_Spherical, Cartesian_to_Cylindrical
    if func_system == C:
        return func(x_mesh, y_mesh, z_mesh, **kwargs)
    elif func_system == P:
        r , p, z = Cartesian_to_Cylindrical(x_mesh, y_mesh, z_mesh)
        polarVector = r*P.i + p*P.j + z*P.k
        ResultMatrix = func(polarVector, **kwargs).to_matrix(P)
        P_to_C_matrix = Matrix([[cos(p), -sin(p), 0], [sin(p), cos(p), 0], [0, 0, 1]])
        return matrix_to_vector(P_to_C_matrix*ResultMatrix, C)
    elif func_system == S:
        r, t, p = Cartesian_to_Spherical(x_mesh, y_mesh, z_mesh)
        sphericalVector = r*S.i + p*S.j + t*S.k
        ResultMatrix = func(sphericalVector, **kwargs).to_matrix(S)
        S_to_C_matrix = Matrix([
            [sin(t)*cos(p), cos(t)*cos(p), -sin(p)],
            [sin(t)*sin(p), cos(t)*sin(p), cos(p)],
            [cos(t), -sin(t), 0]])
        return matrix_to_vector(S_to_C_matrix*ResultMatrix, C)
    else:
        raise ValueError("func_system must be a valid coordinate system: C, P or S.")

def symetric_on_derivative(expr, system, variable=[]):
    if system == C:
        system_variables = [C.x, C.y, C.z]
        for var in system_variables:
            if var in variable:
                expr = change_derivatives_to_constant_by_denominator(expr, var, 0)
    elif system == P:
        system_variables = [P.r, P.phi, P.z]
        for var in system_variables:
            if var  in variable:
                expr = change_derivatives_to_constant_by_denominator(expr, var, 0)
    elif system == S:
        system_variables = [S.r, S.theta, S.phi]
        for var in system_variables:
            if var  in variable:
                expr = change_derivatives_to_constant_by_denominator(expr, var, 0)
    else:
        raise ValueError("system must be a valid coordinate system: C, P or S.")
    return expr

def vector_path_integral(V, cor1, cor2, cor3):
    if not isinstance(V, Vector):
        raise ValueError("V must be a Vector")
    system = V._sys
    if system == C:
        x_int = Integral((V.coeff(C.i)), cor1).doit()
        y_int = Integral((V.coeff(C.j)), cor2).doit()
        z_int = Integral((V.coeff(C.k)), cor3).doit()
        return x_int + y_int + z_int
    elif system == P:
        if isinstance(cor1, tuple):
            r_int = Integral((V.coeff(P.i)), cor1).doit()
            phi_int = Integral((cor1[0]*V.coeff(P.j)), cor2).doit()
            if cor1[1] == cor1[2]:
                phi_int = phi_int.subs(P.r, cor1[0])
        else:
            r_int = 0
            phi_int = Integral((cor1*V.coeff(P.j)), cor2).doit()
            phi_int = phi_int.subs(P.r, cor1)
        z_int = Integral((V.coeff(P.k)), cor3).doit()

        return r_int + phi_int + z_int
    elif system == S:
        r_int= Integral((V.coeff(S.i)), cor1).doit()
        cor1sym = cor1
        cor2sym = cor2
        if isinstance(cor1, tuple):
            cor1sym = cor1[0]
        if isinstance(cor2, tuple):
            cor2sym = cor2[0]
        theta_int = Integral((cor1sym*V.coeff(S.j)), cor2).doit()
        phi_int = Integral((cor1sym*sin(cor2sym)*V.coeff(S.k)), cor3).doit()
        return r_int + theta_int + phi_int
    else:
        raise ValueError("V must be a Vector of a valid coordinate system: C, P or S.")
    
def vector_circle_integral(V, radius):
    if not isinstance(V, Vector):
        raise ValueError("V must be a Vector")
    if radius <= 0:
        raise ValueError("radius must be greater than 0")
    if is_conservative(V):
        print("The vector field is conservative")
        return 0
    system = V._sys
    if system == C:
        t = symbols('t')
        x = radius * cos(t)
        y = radius * sin(t)
        V = V.subs({C.x: x, C.y: y})
        Vi = V.coeff(C.i)*radius*diff(x, t) * C.i
        Vj = V.coeff(C.j)*radius*diff(y, t) * C.j
        Vk = V.coeff(C.k) * C.k
        V = Vi + Vj + Vk
        return vector_path_integral(V,
                                    (t, 0, 2 * pi), (t, 0, 2 * pi), (C.z, 0, 0))
    elif system == P:
        return vector_path_integral(V, radius,
                                    (P.phi, 0, 2*pi), (P.z, 0, 0))
    elif system == S:
        return vector_path_integral(V, radius,
                                    (S.phi, 0, 2*pi), (S.theta, 0, 0))
    else:
        raise ValueError("V must be a Vector of a valid coordinate system: C, P or S.")

