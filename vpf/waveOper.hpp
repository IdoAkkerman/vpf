
// This file is part of the VPF application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
//   __     ______  _____
//   \ \   / /  _ \|  ___|
//    \ \ / /| |_) | |_
//     \ V / |  __/|  _|
//      \_/  |_|   |_|
//
//
// VPF is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------

#ifndef VPF_WAVE_OPERATOR_HPP
#define VPF_WAVE_OPERATOR_HPP

#include "mfem.hpp"

using namespace mfem;

/** After spatial discretization, the wave model can be written as:
 *
 *     d^2u/dt^2 = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass,
 *  and K is the stiffness matrix.
 *
 *  Class WaveOperator represents the right-hand side of the above ODE.
 */
class WaveOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
 //  Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   BilinearForm *K;

   SparseMatrix Mmat, Kmat;
   SparseMatrix *T; // T = M + dt K
   real_t current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + fac0*K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   Coefficient *c2;
   mutable Vector z; // auxiliary vector

public:
   WaveOperator(FiniteElementSpace &f, real_t speed);

   using TimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, Vector &du_dt) const;

   /** Solve the Backward-Euler equation:
       d2udt2 = f(u + fac0*d2udt2,dudt + fac1*d2udt2, t),
       for the unknown d2udt2. */
   using TimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const real_t fac0,
                              const Vector &u,
                              Vector &d2udt2);

   ///
   void SetParameters(const Vector &u);

   virtual ~WaveOperator();
};

#endif
