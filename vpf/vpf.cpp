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
         if (prec[i]) { delete prec[i]; }
      }
   }

};

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE and print info
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   printInfo();

   // Parse command-line options.
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

   args.AddOption(&freesurf_bdr, "-fs", "--freesurf-bdr",
                  "Free-surface boundaries.");
   args.AddOption(&bottom_bdr, "-bot", "--bottom-bdr",
                  "Bottom Boundaries.");
   args.AddOption(&outflow_bdr, "-out", "--outflow-bdr",
                  "Outflow boundaries.");
   args.AddOption(&inflow_bdr, "-in", "--inflow-bdr",
                  "Inflow boundaries.");
   args.AddOption(&master_bdr, "-mbc", "--master-bdr",
                  "Periodic master boundaries.");
   args.AddOption(&slave_bdr, "-sbc", "--slave-bdr",
                  "Periodic slave boundaries.");
   args.AddOption(&lib_file, "-l", "--lib",
                  "Library file for case specific function definitions:\n\t"
                  " - Initial condition\n\t"
                  " - Boundary condition\n\t"
                  " - Forcing\n\t"
                  " - Diffusion\n\t");

   // Solver parameters
   double GMRES_RelTol = 1e-10;
   int    GMRES_MaxIter = 500;

   args.AddOption(&GMRES_RelTol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&GMRES_MaxIter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");

   // Wave parameters
   real_t gravity = 9.81;
   real_t depth  =  10.0;

   real_t wave_angle_min = 0.0;
   real_t wave_angle_max = 180.0;
   int    wave_angle_num = 7;

   real_t wave_length_min = 5.0;
   real_t wave_length_max = 100.0;
   int    wave_length_num = 20;

   args.AddOption(&gravity, "-g", "--gravity",
                  "Gravity parameter (in m/s^2).");

   args.AddOption(&depth, "-d", "--depth",
                  "Depth of the water (in m).");

   args.AddOption(&wave_angle_min, "-wamn", "--wave-angle-min",
                  "Wave angle minimum (in deg).");

   args.AddOption(&wave_angle_max, "-wamx", "--wave-angle-max",
                  "Wave angle maximum (in deg).");

   args.AddOption(&wave_angle_num, "-wan", "--wave-angle-num",
                  "Number of wave angles to compute. Will uniformly "
                  "split the interval [wave-angle-min, wave-angle-max]");

   args.AddOption(&wave_length_min, "-wlmn", "--wave-length-min",
                  "Wave length minimum (in m).");

   args.AddOption(&wave_length_max, "-wlmx", "--wave-length-max",
                  "Wave length maximum (in m).");

   args.AddOption(&wave_length_num, "-wln", "--wave-length-num",
                  "Number of wave lengths to compute. Will uniformly "
                  "split the interval [wave-length-min, wave-length-max]");

   // Solution location
   const char *vis_dir = "solution";

   args.AddOption(&vis_dir, "-vd", "--vis-dir",
                  "Directory for visualization files.\n\t");

   real_t omega = 0.5;
 //  real_t gravity = 9.81;
   real_t cel = 112.0;


   // Parse parameters
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // Read the mesh from the given mesh file.
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
      if (freesurf_bdr.Size()>0) {cout<<"Free-surface = "; freesurf_bdr.Print();}
      if (bottom_bdr.Size()>0)   {cout<<"Bottom       = "; bottom_bdr.Print();}
      if (outflow_bdr.Size()>0)  {cout<<"Outflow      = "; outflow_bdr.Print() ;}
      if (inflow_bdr.Size()>0)   {cout<<"Inflow       = "; inflow_bdr.Print() ;}
      if (master_bdr.Size()>0) {cout<<"Periodic (master) = "; master_bdr.Print();}
      if (slave_bdr.Size()>0) {cout<<"Periodic (slave)  = "; slave_bdr.Print();}
   }

   Array<bool> bnd_flag(pmesh.bdr_attributes.Max()+1);
   bnd_flag = true;
   for (int b = 0; b < pmesh.bdr_attributes.Size(); b++)
   {
      bnd_flag[pmesh.bdr_attributes[b]] = false;
   }

   CheckBoundaries(bnd_flag, freesurf_bdr);
   CheckBoundaries(bnd_flag, bottom_bdr);
   CheckBoundaries(bnd_flag, outflow_bdr);
   CheckBoundaries(bnd_flag, inflow_bdr);
   CheckBoundaries(bnd_flag, master_bdr);
   CheckBoundaries(bnd_flag, slave_bdr);

   MFEM_VERIFY(master_bdr.Size() == master_bdr.Size(),
               "Master-slave count do not match.");
   for (int b = 0; b < bnd_flag.Size(); b++)
   {
      MFEM_VERIFY(bnd_flag[b],
                  "Not all boundaries have a boundary condition set.");
   }

   // Define a finite element space on the mesh.
   FiniteElementCollection* fec = FECollection::NewH1(order, dim, pmesh.IsNURBS());

   ParFiniteElementSpace space(&pmesh, fec, 1,  Ordering::byNODES);
   //, Ordering::byVDIM);
   // ,master_bdr, slave_bdr);

   // Report the degree of freedoms
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
         if ( num_procs > 1)
         {
            mfem::out << "Number of finite element unknowns per partition: \n";
            dof.Print(mfem::out, num_procs);
            dof.Print(mfem::out, num_procs);
         }
      }
   }

   // Get vector offsets
   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = space.GetVSize();
   bOffsets[2] = space.GetVSize();
   bOffsets.PartialSum();

   Array<int> btOffsets(3);
   btOffsets[0] = 0;
   btOffsets[1] = space.TrueVSize();
   btOffsets[2] = space.TrueVSize();
   btOffsets.PartialSum();

   // Define the vectors
   BlockVector x(bOffsets), rhs(bOffsets);
   BlockVector trueX(btOffsets), trueRhs(btOffsets);

   // Define the gridfunctions
   ParGridFunction phi_re(&space);
   ParGridFunction phi_im(&space);

   // Define the visit visualisation output
   VisItDataCollection vdc("step", &pmesh);
   vdc.SetPrefixPath(vis_dir);
   vdc.RegisterField("phi_re", &phi_re);
   vdc.RegisterField("phi_im", &phi_im);

   // Define the paraview visualisation output
   ParaViewDataCollection pdc(vis_dir, &pmesh);
   pdc.SetLevelsOfDetail(order);
   pdc.SetDataFormat(VTKFormat::BINARY);
   pdc.SetHighOrderOutput(true);
   pdc.RegisterField("phi_re", &phi_re);
   pdc.RegisterField("phi_im", &phi_im);

   // Compute the linear forms
   ConstantCoefficient one(1.0);
   ParLinearForm *lform_re(new ParLinearForm);
   lform_re->Update(&space, rhs.GetBlock(0), 0);
   lform_re->AddDomainIntegrator(new DomainLFIntegrator(one));
   lform_re->Assemble();
   lform_re->SyncAliasMemory(rhs);
   lform_re->ParallelAssemble(trueRhs.GetBlock(0));
   trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);

   ConstantCoefficient two(2.0);
   ParLinearForm *lform_im(new ParLinearForm);
   lform_im->Update(&space, rhs.GetBlock(1), 0);
   lform_im->AddDomainIntegrator(new DomainLFIntegrator(two));
   lform_im->Assemble();
   lform_im->SyncAliasMemory(rhs);
   lform_im->ParallelAssemble(trueRhs.GetBlock(1));
   trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);

   //   lform_re->AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
   //   lform_im->AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);

   // Compute the bilinear forms
   ParBilinearForm poisson(&space);
   poisson.AddDomainIntegrator(new DiffusionIntegrator);
   poisson.Assemble();
   poisson.Finalize();

   ConstantCoefficient zero(1e-100);
   ParBilinearForm fs_form(&space);
   fs_form.AddDomainIntegrator(new MassIntegrator(zero));
   fs_form.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one),freesurf_bdr);
   fs_form.Assemble();
   fs_form.Finalize();

   ParBilinearForm out_form(&space);
   out_form.AddDomainIntegrator(new MassIntegrator(zero));
   out_form.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one),outflow_bdr);
   out_form.Assemble();
   out_form.Finalize();

   HypreParMatrix *A    = poisson.ParallelAssemble();
   HypreParMatrix *Afs  = fs_form.ParallelAssemble();
   HypreParMatrix *Aout = out_form.ParallelAssemble();

   // Set Solver
   GMRESSolver solver(MPI_COMM_WORLD);
   solver.SetRelTol(GMRES_RelTol);
   solver.SetAbsTol(1e-10);
   solver.SetMaxIter(GMRES_MaxIter);
   solver.SetPrintLevel(2);

   // Loop over wave conditions
   real_t dwa = (wave_angle_max- wave_angle_min)/(wave_angle_num - 1);
   real_t dwl = (wave_length_max - wave_length_min)/(wave_length_num - 1);

   for (int ia = 0; ia < wave_angle_num; ia++)
   {
      real_t angle = wave_angle_min + ia*dwa;
      for (int il = 0; il < wave_length_num; il++)
      {
         real_t length = wave_length_min + il*dwl;
         real_t k = 2*M_PI/length;
         real_t omega = sqrt(gravity*k*tanh(k*depth));
         real_t celerity = omega/k;
         cout<<"========================================"<<endl;
         cout<<" Wave parameters"<<endl;
         cout<<"========================================"<<endl;
         cout<<" Angle    = "<<angle<<endl;
         cout<<" Length   = "<<length<<endl;
         cout<<" Omega    = "<<omega<<endl;
         cout<<" Celerity = "<<celerity<<endl;

         // Set matrix
         HypreParMatrix Ad(*A);
         Ad.Add(-omega*omega/gravity, *Afs);

         BlockOperator jac(bOffsets);
         jac.SetBlock(0, 0, &Ad);
         jac.SetBlock(0, 1, Aout, omega/celerity);

         jac.SetBlock(1, 0, Aout, -omega/celerity);
         jac.SetBlock(1, 1, &Ad);

         // Preconditioner
         HypreEuclid M(Ad);
         BlockDiagonalPreconditioner precon(bOffsets);
         precon.SetDiagonalBlock(0, &M);
         precon.SetDiagonalBlock(1, &M);

         // Solve
         solver.SetPreconditioner(precon);
         solver.SetOperator(jac);
         solver.Mult(trueRhs,trueX);


         // Visualize solution
         phi_re.Distribute(trueX.GetBlock(0));
         phi_im.Distribute(trueX.GetBlock(1));
         vdc.SetCycle(ia*1000+il);
         vdc.Save();

         pdc.SetCycle(ia*1000+il);
         pdc.Save();

      }
   }
 /*  omega = 0.5;
   for (int l = 0; l < 15; l++)
   {

      omega *=1.1;


      HypreParMatrix Ad(*A);
      Ad.Add(-omega*omega/gravity, *Afs);

      BlockOperator jac(bOffsets);
      jac.SetBlock(0, 0, &Ad);
      jac.SetBlock(0, 1, Aout, omega/cel);

      jac.SetBlock(1, 0, Aout, -omega/cel);
      jac.SetBlock(1, 1, &Ad);

      // Preconditioner
      HypreSmoother M(Ad);
      BlockDiagonalPreconditioner precon(bOffsets);
      precon.SetDiagonalBlock(0, &M);
      precon.SetDiagonalBlock(1, &M);

      // Solver
   //   CGSolver cg(MPI_COMM_WORLD);
   //   cg.SetRelTol(1e-12);
    //  cg.SetMaxIter(2000);
   //   cg.SetPrintLevel(1);
    //  cg.SetPreconditioner(precon);
   //   cg.SetOperator(jac);
   //   cg.Mult(trueRhs,trueX);

         // Solve
         solver.SetPreconditioner(precon);
         solver.SetOperator(jac);
         solver.Mult(trueRhs,trueX);

      // Visualize solution
      phi_re.Distribute(trueX.GetBlock(0));
      phi_im.Distribute(trueX.GetBlock(1));
      vdc.SetCycle(l);
      vdc.Save();

      pdc.SetCycle(l);
      pdc.Save();
   }*/

   // Free the used memory.
   delete fec;

   return 0;
}
