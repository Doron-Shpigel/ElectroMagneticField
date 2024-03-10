import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import sin, cos, acos, atan2, sqrt, exp, pi, symbols, UnevaluatedExpr, latex, Piecewise, Function, solve, Derivative
from sympy.vector import Vector, cross, CoordSys3D, ParametricRegion, ImplicitRegion, vector_integrate, Del, curl, express, divergence, gradient, ParametricIntegral
from sympy.geometry import Point, Circle, Triangle
from sympy import init_printing
from sympy.printing import latex
from sympy import Integral, integrate
from numpy import meshgrid, ravel

def coords():
    #Initial coordinate systems setup
    C = CoordSys3D('C') #(x, y, z)
    P = C.create_new('P', transformation='cylindrical', variable_names = ["r", "phi", "z"]) #r, phi, z: (r*cos(phi), r*sin(phi), z)
    S = C.create_new('S', transformation='spherical') #r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta))
    return C, P, S

C, P, S = coords()

def PrintVector(V):
    if isinstance(V, Vector):
        system = V._sys
        if system == C:
            str_vector_i = rf"\left[" + latex(V.coeff(C.i)) + rf"\right] \hat{{x}}"
            str_vector_j = rf"\left[" + latex(V.coeff(C.j)) + rf"\right] \hat{{y}}"
            str_vector_k = rf"\left[" + latex(V.coeff(C.k)) + rf"\right] \hat{{z}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{x}_{C}}", "x")
            str_vector = str_vector.replace(r"\mathbf{{y}_{C}}", "y")
            str_vector = str_vector.replace(r"\mathbf{{z}_{C}}", "z")
            return str_vector
        elif system == P:
            str_vector_i = rf"\left[" + latex(V.coeff(P.i)) + rf"\right] \hat{{r}}"
            str_vector_j = rf"\left[" + latex(V.coeff(P.j)) + rf"\right] \hat{{\phi}}"
            str_vector_k = rf"\left[" + latex(V.coeff(P.k)) + rf"\right] \hat{{z}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{r}_{P}}", "r")
            str_vector = str_vector.replace(r"\mathbf{{phi}_{P}}", "\phi")
            str_vector = str_vector.replace(r"\mathbf{{z}_{P}}", "z")
            return str_vector
        elif system == S:
            str_vector_i = rf"\left[" + latex(V.coeff(S.i)) + rf"\right] \hat{{r}}"
            str_vector_j = rf"\left[" + latex(V.coeff(S.j)) + rf"\right] \hat{{\theta}}"
            str_vector_k = rf"\left[" + latex(V.coeff(S.k)) + rf"\right] \hat{{\phi}}"
            str_vector =  str_vector_i + "+" + str_vector_j + "+" + str_vector_k
            str_vector = str_vector.replace(r"\mathbf{{r}_{S}}", "r")
            str_vector = str_vector.replace(r"\mathbf{{theta}_{S}}", "\theta")
            str_vector = str_vector.replace(r"\mathbf{{phi}_{S}}", "\phi")
            return str_vector
    else:
        x, y, z = symbols('x y z')
        V = V.subs(C.x, UnevaluatedExpr(x))
        V = V.subs(C.y, UnevaluatedExpr(y))
        V = V.subs(C.z, UnevaluatedExpr(z))
        r, theta, phi = symbols('r theta phi')
        V = V.subs(S.r, UnevaluatedExpr(r))
        V = V.subs(S.theta, UnevaluatedExpr(theta))
        V = V.subs(S.phi, UnevaluatedExpr(phi))
        r, phi, z = symbols('r phi z')
        V = V.subs(P.r, UnevaluatedExpr(r))
        V = V.subs(P.phi, UnevaluatedExpr(phi))
        V = V.subs(P.z, UnevaluatedExpr(z))
        return latex(V)

def NewVector(name, system):
    if system == C:
        a = Function(f"{name}x")(C.x, C.y, C.z)
        b = Function(f"{name}y")(C.x, C.y, C.z)
        c = Function(f"{name}z")(C.x, C.y, C.z)
        return a*C.i + b*C.j + c*C.k
    elif system == P:
        a = Function(f"{name}r")(P.r, P.phi, P.z)
        b = Function(f"{name}phi")(P.r, P.phi, P.z)
        c = Function(f"{name}z")(P.r, P.phi, P.z)
        return a*P.i + b*P.j + c*P.k
    elif system == S:
        a = Function(f"{name}r")(S.r, S.theta, S.phi)
        b = Function(f"{name}theta")(S.r, S.theta, S.phi)
        c = Function(f"{name}phi")(S.r, S.theta, S.phi)
        return a*S.i + b*S.j + c*S.k

class NewVectorFunction:

    def __init__(self, name, system):
        self.name = name
        self.system = system
        if system == C:
            self.i = Function(f"{self.name}x")(C.x, C.y, C.z)
            self.j = Function(f"{self.name}y")(C.x, C.y, C.z)
            self.k = Function(f"{self.name}z")(C.x, C.y, C.z)
            #self.vector = self.i*C.i + self.j*C.j + self.k*C.k
        elif system == P:
            self.i = Function(f"{self.name}r")(P.r, P.phi, P.z)
            self.j = Function(f"{self.name}phi")(P.r, P.phi, P.z)
            self.k = Function(f"{self.name}z")(P.r, P.phi, P.z)
            #self.vector = self.i*P.i + self.j*P.j + self.k*P.k
        elif system == S:
            self.i = Function(f"{self.name}r")(S.r, S.theta, S.phi)
            self.j = Function(f"{self.name}theta")(S.r, S.theta, S.phi)
            self.k = Function(f"{self.name}phi")(S.r, S.theta, S.phi)
            #self.vector = self.i*S.i + self.j*S.j + self.k*S.k
        self.update_vector()

    def update_vector(self):
        self.vector = self.i*S.i + self.j*S.j + self.k*S.k

    def __call__(self):
        self.update_vector()
        return self.vector
    def __str__(self):
        self.update_vector()
        return PrintVector(self.vector)
    def __repr__(self):
        self.update_vector()
        return self.vector

def Cartesian_to_Spherical(x, y, z):
    r = sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0, 0, 0
    theta = acos(z/r)
    phi = atan2(y, x)
    return r, theta, phi

def Cartesian_to_Cylindrical(x, y, z):
    r = sqrt(x**2 + y**2)
    phi = atan2(y, x)
    return r, phi, z

class VectorMesh:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.func = None
        self.x_mesh, self.y_mesh, self.z_mesh = meshgrid(x, y, z)
        self.data = self.data(self.x_mesh, self.y_mesh, self.z_mesh)

    class data():
        def __init__(self, x_mesh = None, y_mesh = None, z_mesh = None):
            self.x = x_mesh.ravel()
            self.y = y_mesh.ravel()
            self.z = z_mesh.ravel()

    def field(self, func_system, func, **kwargs):
        self.func = func
        self.u_mesh, self.v_mesh, self.w_mesh = meshgrid(self.x, self.y, self.z)
        from src.SpecialOperators import XYZ_to_Function
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                for k in range(len(self.z)):
                    result_vector = XYZ_to_Function(self.x_mesh[i, j, k], self.y_mesh[i, j, k], self.z_mesh[i, j, k], func_system, func, **kwargs)
                    self.u_mesh[i][j][k] = result_vector.coeff(C.i)
                    self.v_mesh[i][j][k] = result_vector.coeff(C.j)
                    self.w_mesh[i][j][k] = result_vector.coeff(C.k)
        self.data.u = self.u_mesh.ravel()
        self.data.v = self.v_mesh.ravel()
        self.data.w = self.w_mesh.ravel()

    def offset_field(self, origin=[0,0,0]):
        self.data.x += origin[0]
        self.data.y += origin[1]
        self.data.z += origin[2]

    def plotlyConeData(self, title = None, **kwargs):
        if self.func == None:
            raise ValueError("You must set a function to the field before plotting it.")
        import plotly.graph_objects as go
        data = go.Cone(x=self.data.x, y=self.data.y, z=self.data.z,
                u=self.data.u, v=self.data.v, w=self.data.w,
                colorscale='Inferno', colorbar=dict(title=title),
                sizemode="absolute", sizeref=0.1)
        return data

def cylinder(origin = [0,0,0], cylinder_radius = 1, cylinder_height=1,  **kwargs):
    if not hasattr(cylinder, "N"):
        cylinder.N = 0
    from numpy import pi, linspace, meshgrid, cos, sin
    cylinder_height = linspace(0, cylinder_height, 100)
    phi = linspace(0, 2*pi, 100)
    phi_mesh, height_mesh = meshgrid(phi, cylinder_height)
    x_mesh = cylinder_radius * cos(phi_mesh) + origin[0]
    y_mesh = cylinder_radius * sin(phi_mesh) + origin[1]
    z_mesh = height_mesh + origin[2]
    import plotly.graph_objects as go
    surface = go.Surface(x=x_mesh, y=y_mesh, z=z_mesh,
                name=f"cylinder_{cylinder.N}",
                colorscale=[[0, 'lightblue'], [1, 'blue']], showscale=False,
                showlegend=False, **kwargs)
    cylinder.N += 1
    return surface
