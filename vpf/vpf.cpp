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

#include "mfem.hpp"
//#include "coefficients.hpp"
//#include "weakform.hpp"
//#include "evolution.hpp"
//#include "precon.hpp"
//#include "monitor.hpp"

#include <sys/stat.h>

using namespace std;
using namespace mfem;

extern void printInfo();
extern void line(int len);

void CheckBoundaries(Array<bool> &bnd_flags,
                     Array<int> &bc_bnds)
{
   int amax = bnd_flags.Size();
   // Check strong boundaries
   for (int b = 0; b < bc_bnds.Size(); b++)
   {
      int bnd = bc_bnds[b];
      if ( bnd < 0 || bnd > amax )
      {
         mfem_error("Boundary out of range.");
      }
      if (bnd_flags[bnd])
      {
         mfem_error("Boundary specified more then once.");
      }
      bnd_flags[bnd] = true;
   }
}

// Custom block preconditioner for the Jacobian
class JacobianPreconditioner : public
   BlockLowerTriangularPreconditioner
{
protected:
   Array<Solver *> prec;

public:
   /// Constructor
   JacobianPreconditioner(Array<int> &offsets)
      : BlockLowerTriangularPreconditioner (offsets), prec(offsets.Size()-1)
   { prec = nullptr;};

   /// SetPreconditioners
   void SetPreconditioner(int i, Solver *pc)
   { prec[i] = pc; };

   /// Set the diagonal and off-diagonal operators
   virtual void SetOperator(const Operator &op)
   {
      BlockOperator *jacobian = (BlockOperator *) &op;

      for (int i = 0; i < prec.Size(); ++i)
      {
         if (prec[i])
         {
            prec[i]->SetOperator(jacobian->GetBlock(i,i));
            SetDiagonalBlock(i, prec[i]);
         }
         for (int j = i+1; j < prec.Size(); ++j)
         {
            SetBlock(j,i, &jacobian->GetBlock(j,i));
         }
      }
   }

   // Destructor
   virtual ~JacobianPreconditioner()
   {
      for (int i = 0; i < prec.Size(); ++i)
      {
         if (prec[i]) delete prec[i];
      }
   }

};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE and print info
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   printInfo();

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);

   // Mesh and discretization parameters
   const char *mesh_file = "../../mfem/data/inline-quad.mesh";
   const char *ref_file  = "";
   int order = 1;
   int ref_levels = 0;
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order isoparametric space.");

   // Problem parameters
   Array<int> freesurf_bdr;
   Array<int> bottom_bdr;
   Array<int> outflow_bdr;
   Array<int> inflow_bdr;

   Array<int> master_bdr;
   Array<int> slave_bdr;

   const char *lib_file = "libfun.so";

 /*  args.AddOption(&strong_bdr, "-sbc", "--strong-bdr",
                  "Boundaries where Dirichelet BCs are enforced strongly.");
   args.AddOption(&weak_bdr, "-wbc", "--weak-bdr",
                  "Boundaries where Dirichelet BCs are enforced weakly.");
   args.AddOption(&outflow_bdr, "-out", "--outflow-bdr",
                  "Outflow boundaries.");
   args.AddOption(&suction_bdr, "-suc", "--suction-bdr",
                  "Suction boundaries.");
   args.AddOption(&blowing_bdr, "-blow", "--blowing-bdr",
                  "Blowing boundaries.");
   args.AddOption(&master_bdr, "-mbc", "--master-bdr",
                  "Periodic master boundaries.");*/
   args.AddOption(&slave_bdr, "-sbc", "--slave-bdr",
                  "Periodic slave boundaries.");
   args.AddOption(&lib_file, "-l", "--lib",
                  "Library file for case specific function definitions:\n\t"
                  " - Initial condition\n\t"
                  " - Boundary condition\n\t"
                  " - Forcing\n\t"
                  " - Diffusion\n\t");

   // Time stepping params
//   Array<int> master_bdr;
//   Array<int> slave_bdr;

  // int ode_solver_type = 35;
  //int ode_solver_type = 35;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   real_t dt_max = 1.0;
   real_t dt_min = 0.0001;

   real_t cfl_target = 2.0;
   real_t dt_gain = -1.0;

 //  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
 //                 ODESolver::ImplicitTypes.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--dt",
                  "Time step.");
   args.AddOption(&dt_min, "-dtmn", "--dt-min",
                  "Minimum time step size.");
   args.AddOption(&dt_max, "-dtmx", "--dt-max",
                  "Maximum time step size.");
   args.AddOption(&cfl_target, "-cfl", "--cfl-target",
                  "CFL target.");
   args.AddOption(&dt_gain, "-dtg", "--dt-gain",
                  "Gain coefficient for time step adjustment.");

   // Solver parameters
   double GMRES_RelTol = 1e-3;
   int    GMRES_MaxIter = 500;

   args.AddOption(&GMRES_RelTol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&GMRES_MaxIter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");

   // Solution input/output params
   bool restart = false;
   int restart_interval = -1;
   real_t dt_vis = 10*dt;
   const char *vis_dir = "solution";
   args.AddOption(&restart, "-rs", "--restart", "-f", "--fresh",
                  "Restart from solution.");
   args.AddOption(&restart_interval, "-ri", "--restart-interval",
                  "Interval between archieved time steps.\n\t"
                  "For negative values output is skipped.");
   args.AddOption(&dt_vis, "-dtv", "--dt_vis",
                  "Time interval between visualization points.");
   args.AddOption(&vis_dir, "-vd", "--vis-dir",
                  "Directory for visualization files.\n\t");


   // Parse parameters
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // 3. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine mesh
   {
      if (mesh.NURBSext && (strlen(ref_file) != 0))
      {
         mesh.RefineNURBSFromFile(ref_file);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
      if (Mpi::Root()) { mesh.PrintInfo(); }
   }

   // Partition mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Boundary conditions
   if (Mpi::Root())
   {
  /*    if (strong_bdr.Size()>0) {cout<<"Strong  = "; strong_bdr.Print();}
      if (weak_bdr.Size()>0) {cout<<"Weak    = "; weak_bdr.Print();}
      if (outflow_bdr.Size()>0){ cout<<"Outflow = "; outflow_bdr.Print() ;}
      if (suction_bdr.Size()>0){ cout<<"Suction = "; suction_bdr.Print() ;}
      if (blowing_bdr.Size()>0){ cout<<"Blowing = "; blowing_bdr.Print() ;}*/
      if (master_bdr.Size()>0) {cout<<"Periodic (master) = "; master_bdr.Print();}
      if (slave_bdr.Size()>0) {cout<<"Periodic (slave)  = "; slave_bdr.Print();}
   }

   Array<bool> bnd_flag(pmesh.bdr_attributes.Max()+1);
   bnd_flag = true;
   for (int b = 0; b < pmesh.bdr_attributes.Size(); b++)
   {
      bnd_flag[pmesh.bdr_attributes[b]] = false;
   }
  /* CheckBoundaries(bnd_flag, strong_bdr);
   CheckBoundaries(bnd_flag, weak_bdr);
   CheckBoundaries(bnd_flag, outflow_bdr);
   CheckBoundaries(bnd_flag, suction_bdr);
   CheckBoundaries(bnd_flag, blowing_bdr);*/
   CheckBoundaries(bnd_flag, master_bdr);
   CheckBoundaries(bnd_flag, slave_bdr);
/*
   MFEM_VERIFY(master_bdr.Size() == master_bdr.Size(),
               "Master-slave count do not match.");
   for (int b = 0; b < bnd_flag.Size(); b++)
   {
      MFEM_VERIFY(bnd_flag[b],
                 "Not all boundaries have a boundary condition set.");
   }
*/
   // Select the time integrator
 //  unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
 //  int nstate = ode_solver->GetState() ? ode_solver->GetState()->MaxSize() : 0;

//   if (nstate > 1 && ( restart || restart_interval > 0 ))
//   {
 //     mfem_error("RBVMS restart not available for this time integrator \n"
 //                "Time integrator can have a maximum of one statevector.");
//   }

   // 4. Define a finite element space on the mesh.
   FiniteElementCollection* fec = FECollection::NewH1(order, dim, pmesh.IsNURBS());

   ParFiniteElementSpace space(&pmesh, fec, 1,  Ordering::byNODES);
                                         //, Ordering::byVDIM);
                                        // ,master_bdr, slave_bdr);

   // Report the degree of freedoms used
   {
      Array<int> tdof(num_procs),dof(num_procs);
      tdof = 0;
      tdof[myid] = space.TrueVSize();
      MPI_Reduce(tdof.GetData(), dof.GetData(), num_procs,
                 MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);


      int dof_t = space.GlobalTrueVSize();
      if (Mpi::Root())
      {
         mfem::out << "Number of finite element unknowns: ";
         mfem::out << dof_t << " " <<dof_t  << endl;
         mfem::out << "Number of finite element unknowns per partition: \n";
         dof.Print(mfem::out, num_procs);
         dof.Print(mfem::out, num_procs);
      }
   }

   // 5. Define the time stepping algorithm
   // Get vector offsets
   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = space.TrueVSize();
   bOffsets[2] = space.TrueVSize();
   bOffsets.PartialSum();

   // Set up the preconditioner
   JacobianPreconditioner jac_prec(bOffsets);
   Solver* pc_mom = nullptr;
   Solver* pc_cont= nullptr;

   HypreSmoother* hs_mom = new HypreSmoother();
   HypreILU* ilu_cont = new HypreILU();

   pc_mom = hs_mom;
   pc_cont = ilu_cont;

   jac_prec.SetPreconditioner(0, pc_mom);
   jac_prec.SetPreconditioner(1, pc_cont);

   // Set up the Jacobian solver

   FGMRESSolver j_gmres(MPI_COMM_WORLD);
   j_gmres.iterative_mode = false;
   j_gmres.SetRelTol(GMRES_RelTol);
   j_gmres.SetAbsTol(1e-12);
   j_gmres.SetMaxIter(GMRES_MaxIter);
   j_gmres.SetPrintLevel(1);
   j_gmres.SetPreconditioner(jac_prec);

   // 6. Define the solution vector, grid function and output

   // Define the gridfunctions
   ParGridFunction phi_re_gf(&space);
   ParGridFunction phi_im_gf(&space);

   // Define the visualisation output
   VisItDataCollection vdc("step", &pmesh);
   vdc.SetPrefixPath(vis_dir);
   vdc.RegisterField("phi_re", &phi_re_gf);
   vdc.RegisterField("phi_im", &phi_im_gf);

   vdc.SetCycle(0);
   // vdc.SetTime(t);
   vdc.Save();

   // Get the start vector(s) from file -- or from function
   // 7. Actual time integration


   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&space);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParBilinearForm a(&space);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();


   Array<int> boundary_dofs;
   space.GetBoundaryTrueDofs(boundary_dofs);

   boundary_dofs.Print();
   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, phi_re_gf, b, A, X, B);

   // 11. Solve the system using PCG with hypre's BoomerAMG preconditioner.
   HypreBoomerAMG M(A);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, phi_re_gf);



 //  ParLinearForm b(&space);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   a.FormLinearSystem(boundary_dofs, phi_im_gf, b, A, X, B);

   // 11. Solve the system using PCG with hypre's BoomerAMG preconditioner.
  // HypreBoomerAMG M(A);
  // CGSolver cg(MPI_COMM_WORLD);
  // cg.SetRelTol(1e-12);
  // cg.SetMaxIter(2000);
  // cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, phi_im_gf);


         // Actually write to file
         vdc.SetCycle(1);
        // vdc.SetTime(t);
         vdc.Save();









   // 8. Free the used memory.
   delete fec;

   return 0;
}
