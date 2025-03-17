
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

#include "waveOper.hpp"


WaveOperator::WaveOperator(FiniteElementSpace &f, real_t speed)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0),
     fespace(f), M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
   // Assemble Laplace matrix
   c2 = new ConstantCoefficient(speed*speed);
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(*c2));
   K->Assemble();

   // Assemble Mass matrix

   Array<int> fs_bdr(fespace.GetMesh()->bdr_attributes.Max());
   fs_bdr = 0; fs_bdr[2] = 1;

 fs_bdr.Print(std::cout,88);
   M = new BilinearForm(&fespace);
   M->AddBoundaryIntegrator(new MassIntegrator(), fs_bdr);
   M->Assemble();

   // Apply Bcs
 //  fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
 //  K->FormSystemMatrix(ess_tdof_list, Kmat);
 //  M->FormSystemMatrix(ess_tdof_list, Mmat);

   // Configure preconditioner
   const real_t rel_tol = 1e-8;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   // Configure solver
   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);
}

void WaveOperator::Mult(const Vector &u, Vector &dudt)  const
{
   // Compute:
   //    d2udt2 = M^{-1}*-K(u)
   // for d2udt2
   K->FullMult(u, z);
   z.Neg(); // z = -z
//   z.SetSubVector(ess_tdof_list, 0.0);
   M_solver.Mult(z, dudt);
 //  d2udt2.SetSubVector(ess_tdof_list, 0.0);
}

void WaveOperator::ImplicitSolve(const real_t fac0,
                                 const Vector &u,
                                 Vector &dudt)
{
   // Solve the equation:
   //    d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
   // for d2udt2
   if (!T)
   {
      T = Add(1.0, Mmat, fac0, Kmat);
      T_solver.SetOperator(*T);
   }
   K->FullMult(u, z);
   z.Neg();
 //  z.SetSubVector(ess_tdof_list, 0.0);
   T_solver.Mult(z, dudt);
  // d2udt2.SetSubVector(ess_tdof_list, 0.0);
}

void WaveOperator::SetParameters(const Vector &u)
{
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

WaveOperator::~WaveOperator()
{
   delete T;
   delete M;
   delete K;
   delete c2;
}


