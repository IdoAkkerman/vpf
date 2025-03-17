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
#include "waveOper.hpp"
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

void GetBoundaryMarkers(Array<int> &bnd_flags,
                        Array<int> &bnd_attr,
                        Array<int> &bc_bnds)
{
   int amax = bnd_flags.Size();
   bnd_flags = 0;
   // Check strong boundaries
   for (int b = 0; b < bc_bnds.Size(); b++)
   {
      int bnd = bc_bnds[b];
      if ( bnd < 0 || bnd > amax )
      {
         mfem_error("Boundary out of range.");
      }
      bnd_flags[bnd-1] = 1;
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
   const char *mesh_file = "../../mfem/data/inline-hex.mesh";
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

   // Wave parameters
   real_t gravity = 9.81;

   real_t wave_angle_min = 0.0;
   real_t wave_angle_max = 90.0;
   int    wave_angle_num = 4;

   real_t wave_length_min = 20.0;
   real_t wave_length_max = 100.0;
   int    wave_length_num = 5;

   args.AddOption(&gravity, "-g", "--gravity",
                  "Gravity parameter (in m/s^2).");

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

   // Time stepping params
   int ode_solver_type = 45;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   real_t dt_max = 1.0;
   real_t dt_min = 0.0001;

   real_t cfl_target = 2.0;
   real_t dt_gain = -1.0;

   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::ImplicitTypes.c_str());
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
   double Newton_RelTol = 1e-3;
   int    Newton_MaxIter = 10;

   args.AddOption(&GMRES_RelTol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&GMRES_MaxIter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");
   args.AddOption(&Newton_RelTol, "-nt", "--newton-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&Newton_MaxIter, "-ni", "--newton-itermax",
                  "Maximum iteration count for the Newton solver.");

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

   // Select time integrator
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   int nstate = ode_solver->GetState() ? ode_solver->GetState()->MaxSize() : 0;

   if (nstate > 1 && ( restart || restart_interval > 0 ))
   {
      mfem_error("RBVMS restart not available for this time integrator \n"
                 "Time integrator can have a maximum of one statevector.");
   }

   // Read mesh
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   Vector min(dim), max(dim);
   mesh.GetBoundingBox(min, max);
   real_t depth = max(dim-1) - min(dim-1);

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


   // Define a finite element space
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
   trueX = 0.0;
   trueRhs = 0.0;

   BlockVector xp(bOffsets);
   BlockVector dxp(bOffsets);
   BlockVector xp0(bOffsets);
   BlockVector xpi(bOffsets);

   // Define the gridfunctions
   ParGridFunction phi(&space);
   ParGridFunction eta(&space);

   Array<ParGridFunction*> dphi(nstate);
   Array<ParGridFunction*> deta(nstate);

   for (int i = 0; i < nstate; i++)
   {
      dphi[i] = new ParGridFunction(&space);
      deta[i] = new ParGridFunction(&space);
   }

   // Define the visit visualisation output
   VisItDataCollection vdc("step", &pmesh);
   vdc.SetPrefixPath(vis_dir);
   vdc.RegisterField("phi", &phi);
   vdc.RegisterField("eta", &eta);

   // Define the paraview visualisation output
   ParaViewDataCollection pdc(vis_dir, &pmesh);
   pdc.SetLevelsOfDetail(order);
   pdc.SetDataFormat(VTKFormat::BINARY);
   pdc.SetHighOrderOutput(true);
   pdc.RegisterField("phi", &phi);
   pdc.RegisterField("eta", &eta);

   // Define the restart output
   VisItDataCollection rdc("step", &pmesh);
   rdc.SetPrefixPath("restart");
   rdc.SetPrecision(18);

   // Get the start vector(s) from file -- or from function
   real_t t;
   int si, ri, vi;
   struct stat info;
   if (restart && stat("restart/step.dat", &info) == 0)
   {
      // Read
      if (Mpi::Root())
      {
         real_t dtr;
         std::ifstream in("restart/step.dat", std::ifstream::in);
         in>>t>>si>>ri>>vi;
         in>>dtr;
         in.close();
         cout<<"Restarting from step "<<ri-1<<endl;
         if (dt_gain > 0) { dt = dtr; }
      }
      // Synchronize
      MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&si, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ri, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&vi, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Open data files
      rdc.Load(ri-1);

      phi = *rdc.GetField("phi");
      eta = *rdc.GetField("eta");

      phi.GetTrueDofs(xp.GetBlock(0));
      eta.GetTrueDofs(xp.GetBlock(1));

      if (nstate == 1)
      {
         *dphi[0] = *rdc.GetField("dphi");
         *deta[0] = *rdc.GetField("deta");

         dphi[0]->GetTrueDofs(dxp.GetBlock(0));
         deta[0]->GetTrueDofs(dxp.GetBlock(1));

         ode_solver->GetState()->Append(dxp);
      }
   }
   else
   {
      // Define initial condition from file
      t = 0.0; si = 0; ri = 1; vi = 1;
  //    LibVectorCoefficient sol(dim, lib_file, "sol_u");
//      sol.SetTime(-1.0);
   //   phi = 0;//.ProjectCoefficient(sol);
      phi = 0.0;
      eta = 0.0;

      phi.GetTrueDofs(xp.GetBlock(0));
      eta.GetTrueDofs(xp.GetBlock(1));

      // Visualize initial condition
      vdc.SetCycle(0);
      vdc.SetTime(0.0);
      vdc.Save();

      // Define the restart writer
      rdc.RegisterField("phi", &phi);
      rdc.RegisterField("eta", &eta);
      if (nstate == 1)
      {
         rdc.RegisterField("dphi", dphi[0]);
         rdc.RegisterField("deta", deta[0]);
      }
   }

   ///
   real_t speed = 1.0;
   WaveOperator oper(space, speed);
   ode_solver->Init(oper);


   // Boundary conditions
   if (Mpi::Root())
   {
      if (freesurf_bdr.Size()>0) {cout<<"Free-surface = "; freesurf_bdr.Print();}
      if (bottom_bdr.Size()>0)   {cout<<"Bottom       = "; bottom_bdr.Print();}
      if (outflow_bdr.Size()>0)  {cout<<"Outflow      = "; outflow_bdr.Print();}
      if (inflow_bdr.Size()>0)   {cout<<"Inflow       = "; inflow_bdr.Print();}
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

/*   MFEM_VERIFY(master_bdr.Size() == master_bdr.Size(),
               "Master-slave count do not match.");
   for (int b = 0; b < bnd_flag.Size(); b++)
   {
      MFEM_VERIFY(bnd_flag[b],
                  "Not all boundaries have a boundary condition set.");
   }*/



   // 7. Actual time integration

   // Open output file
   std::ofstream os;
   if (Mpi::Root())
   {
      std::ostringstream filename;
      filename << "output_"<<std::setw(6)<<setfill('0')<<si<< ".dat";
      os.open(filename.str().c_str());

      // Header
      char dimName[] = "xyz";
      int i = 6;
      os <<"# 1: step"<<"\t"<<"2: time"<<"\t"<<"3: dt"<<"\t"
         <<"4: cfl"<<"\t"<<"5: outflow"<<"\t";

      for (int b=0; b<pmesh.bdr_attributes.Size(); ++b)
      {
         int bnd = pmesh.bdr_attributes[b];
         for (int v=0; v<dim; ++v)
         {
            std::ostringstream forcename;
            forcename <<i++<<": F"<<dimName[v]<<"_"<<bnd;
            os<<forcename.str()<<"\t";
         }
      }
      os<<endl;
   }

   // Loop till final time reached
   while (t < t_final)
   {
      // Print header
      if (Mpi::Root())
      {
         line(80);
         cout<<std::defaultfloat<<std::setprecision(4);
         cout<<" step = " << si << endl;
         cout<<"   dt = " << dt << endl;
         cout<<std::defaultfloat<<std::setprecision(6);;
         cout<<" time = [" << t << ", " << t+dt <<"]"<< endl;
         cout<<std::defaultfloat<<std::setprecision(4);
         line(80);
      }

      // Actual time step
      xp0 = xp;
      ode_solver->Step(xp, t, dt);
      si++;

      // Postprocess solution
    //  real_t cfl = evo.GetCFL();
    //  real_t outflow = evo.GetOutflow();
   //   DenseMatrix bdrForce = evo.GetForce();
      if (Mpi::Root())
      {
         // Print to file
       /*  int nbdr = pmesh.bdr_attributes.Size();
         os << std::setw(10);
         os << si<<"\t"<<t<<"\t"<<dt<<"\t"<<cfl<<"\t"<<outflow<<"\t";
         for (int b=0; b<nbdr; ++b)
         {
            int bnd = pmesh.bdr_attributes[b];
            for (int v=0; v<dim; ++v)
            {
               os<<bdrForce(bnd-1,v)<<"\t";
            }
         }
         os<<"\n"<< std::flush;*/

         // Print line lambda function
         auto pline = [](int len)
         {
            cout<<" +";
            for (int b=0; b<len; ++b) { cout<<"-"; }
            cout<<"+\n";
         };

         // Print boundary header
         /*cout<<"\n";
         pline(10+13*nbdr);
         cout<<" | Boundary | ";
         for (int b=0; b<nbdr; ++b)
         {
            cout<<std::setw(10)<<pmesh.bdr_attributes[b]<<" | ";
         }
         cout<<"\n";
         pline(10+13*nbdr);

         // Print actual forces
         char dimName[] = "xyz";
         for (int v=0; v<dim; ++v)
         {
            cout<<" | Force "<<dimName[v]<<"  | ";
            for (int b=0; b<nbdr; ++b)
            {
               int bnd = pmesh.bdr_attributes[b];
               cout<<std::defaultfloat<<std::setprecision(4)<<std::setw(10);
               cout<<bdrForce(bnd-1,v)<<" | ";
            }
            cout<<"\n";
         }
         pline(10+13*nbdr);
         cout<<"\n"<<std::flush;*/
      }

      // Write visualization files
      while (t >= dt_vis*vi)
      {
         // Interpolate solution
         real_t fac = (t-dt_vis*vi)/dt;

         // Report to screen
         if (Mpi::Root())
         {
            line(80);
            cout << "Visit output: " <<vi << endl;
            cout << "        Time: " <<t-dt<<" "<<t-fac*dt<<" "<<t<<endl;
            line(80);
         }

         // Copy solution in grid functions
         add (fac, xp0.GetBlock(0),(1.0-fac), xp.GetBlock(0), xpi.GetBlock(0));
         phi.Distribute(xpi.GetBlock(0));

         add (-1.0/dt, xp0.GetBlock(1), 1.0/dt, xp.GetBlock(1), xpi.GetBlock(1));
         eta.Distribute(xpi.GetBlock(1));

         // Actually write to file
         vdc.SetCycle(vi);
         vdc.SetTime(dt_vis*vi);
         vdc.Save();
         vi++;
      }

      // Change time step
      //real_t dt0 = dt;
      //if ((dt_gain > 0))
     // {
      //   dt *= pow(cfl_target/cfl, dt_gain);
       //  dt = min(dt, dt_max);
        // dt = max(dt, dt_min);
    //  }

      // Print cfl and dt to screen
      if (Mpi::Root())
      {
         line(80);
       //  cout<<" outflow = "<<outflow<<endl;
      //   cout<<" cfl = "<<cfl<<endl;
         cout<<" dt  = "<<dt<<" --> "<<dt<<endl;
         line(80);
      }

      // Write restart files
      if (restart_interval > 0 && si%restart_interval == 0)
      {
         // Report to screen
         if (Mpi::Root())
         {
            line(80);
            cout << "Restart output:" << ri << endl;
            line(80);
         }

         // Copy solution in grid functions
         phi.Distribute(xp.GetBlock(0));
         eta.Distribute(xp.GetBlock(1));

         if (nstate == 1)
         {
            ode_solver->GetState()->Get(0,dxp);
            dphi[0]->Distribute(dxp.GetBlock(0));
            deta[0]->Distribute(dxp.GetBlock(1));
         }

         // Actually write to file
         rdc.SetCycle(ri);
         rdc.SetTime(t);
         rdc.Save();
         ri++;

         // print meta file
         if (Mpi::Root())
         {
            std::ofstream step("restart/step.dat", std::ifstream::out);
            step<<t<<"\t"<<si<<"\t"<<ri<<"\t"<<vi<<endl;
            step<<dt<<endl;
            step.close();
         }
      }

      if (Mpi::Root()) { cout<<endl<<endl; }
   }
   os.close();











   return 0;
}
