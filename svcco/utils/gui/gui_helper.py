import tkinter as tk
from tkinter import ttk

class OptionsEntry:
    def __init__(self):
        main = tk.Tk()
        main.title("Simulation Input Parameters")
        main.geometry("400x500")

        rows = 20
        cols = 2

        main.rowconfigure(rows,weight=1)
        main.columnconfigure(cols,weight=1)

        nb = ttk.Notebook(main)
        nb.grid(row=1,column=0,columnspan=20,sticky="NESW")
        nb.enable_traversal()

        geometry_tab = ttk.Frame(nb)
        nb.add(geometry_tab,text="Geometry")

        lofting_tab = ttk.Frame(nb)
        nb.add(lofting_tab,text="Lofting")

        modeling_tab = ttk.Frame(nb)
        nb.add(modeling_tab,text="Modeling")

        meshing_tab = ttk.Frame(nb)
        nb.add(meshing_tab,text="Meshing")

        simulation_tab = ttk.Frame(nb)
        nb.add(simulation_tab,text="Simulation")

        #Options for Geometry Tab
        geom_label1 = ttk.Label(geometry_tab,text="Number of Sample Points")
        geom_label1.grid(row=1,column=0,sticky="W")

        geom_entry1 = ttk.Entry(geometry_tab)
        geom_entry1.insert(0,"50")
        geom_entry1.grid(row=1,column=1,sticky="E")

        geom_label2 = ttk.Label(geometry_tab,text="Use Distance Alignment")
        geom_label2.grid(row=2,column=0,sticky="W")

        geom_entry2 = tk.StringVar()
        geom_entry2.set("False")
        geom_entry2_options = ["False","True"]
        geom_entry2_drop = tk.OptionMenu(geometry_tab,geom_entry2,*geom_entry2_options)
        geom_entry2_drop.grid(row=2,column=1,sticky="E")

        # Options for Lofting Tab
        loft_label1 = ttk.Label(lofting_tab,text="U Knot Span Type")
        loft_label1.grid(row=1,column=0,sticky="W")

        loft_entry1 = tk.StringVar()
        loft_entry1.set("derivative")
        loft_entry1_options = ['derivative','equal','average']
        loft_entry1_drop = tk.OptionMenu(lofting_tab,loft_entry1,*loft_entry1_options)
        loft_entry1_drop.grid(row=1,column=1,sticky="E")

        loft_label2 = ttk.Label(lofting_tab,text="U Parametric Span Type")
        loft_label2.grid(row=2,column=0,sticky="W")

        loft_entry2 = tk.StringVar()
        loft_entry2.set("centripetal")
        loft_entry2_options = ['centripetal','chord','equal']
        loft_entry2_drop = tk.OptionMenu(lofting_tab,loft_entry2,*loft_entry2_options)
        loft_entry2_drop.grid(row=2,column=1,sticky="E")

        loft_label3 = ttk.Label(lofting_tab,text="U Degree")
        loft_label3.grid(row=3,column=0,sticky="W")

        loft_entry3 = ttk.Entry(lofting_tab)
        loft_entry3.insert(0,"2")
        loft_entry3.grid(row=3,column=1,sticky="E")

        loft_label4 = ttk.Label(lofting_tab,text="V Knot Span Type")
        loft_label4.grid(row=4,column=0,sticky="W")

        loft_entry4 = tk.StringVar()
        loft_entry4.set("average")
        loft_entry4_options = ['derivative','equal','average']
        loft_entry4_drop = tk.OptionMenu(lofting_tab,loft_entry4,*loft_entry4_options)
        loft_entry4_drop.grid(row=4,column=1,sticky="E")

        loft_label5 = ttk.Label(lofting_tab,text="V Parametric Span Type")
        loft_label5.grid(row=5,column=0,sticky="W")

        loft_entry5 = tk.StringVar()
        loft_entry5.set("chord")
        loft_entry5_options = ['centripetal','chord','equal']
        loft_entry5_drop = tk.OptionMenu(lofting_tab,loft_entry5,*loft_entry5_options)
        loft_entry5_drop.grid(row=5,column=1,sticky="E")

        loft_label6 = ttk.Label(lofting_tab,text="V Degree")
        loft_label6.grid(row=6,column=0,sticky="W")

        loft_entry6 = ttk.Entry(lofting_tab)
        loft_entry6.insert(0,"2")
        loft_entry6.grid(row=6,column=1,sticky="E")

        loft_label7 = ttk.Label(lofting_tab,text="Boundary Face Angle")
        loft_label7.grid(row=7,column=0,sticky="W")

        loft_entry7 = ttk.Entry(lofting_tab)
        loft_entry7.insert(0,"45")
        loft_entry7.grid(row=7,column=1,sticky="E")

        #Options for Modeling Tab
        model_label1 = ttk.Label(modeling_tab,text="Minimum Number of Face Cells")
        model_label1.grid(row=1,column=0,sticky="W")

        model_entry1 = ttk.Entry(modeling_tab)
        model_entry1.insert(0,"200")
        model_entry1.grid(row=1,column=1,sticky="E")

        model_label2 = ttk.Label(modeling_tab,text="hmin")
        model_label2.grid(row=2,column=0,sticky="W")

        model_entry2 = ttk.Entry(modeling_tab)
        model_entry2.insert(0,"0.02")
        model_entry2.grid(row=2,column=1,sticky="E")

        model_label3 = ttk.Label(modeling_tab,text="hmax")
        model_label3.grid(row=3,column=0,sticky="W")

        model_entry3 = ttk.Entry(modeling_tab)
        model_entry3.insert(0,"0.02")
        model_entry3.grid(row=3,column=1,sticky="E")

        model_label4 = ttk.Label(modeling_tab,text="Boundary Face Angle")
        model_label4.grid(row=4,column=0,sticky="W")

        model_entry4 = ttk.Entry(modeling_tab)
        model_entry4.insert(0,"45")
        model_entry4.grid(row=4,column=1,sticky="E")

        #Options for Meshing Tab

        mesh_label1 = ttk.Label(meshing_tab,text="Global Edge Size")
        mesh_label1.grid(row=1,column=0,sticky="W")

        mesh_entry1 = ttk.Entry(meshing_tab)
        mesh_entry1.insert(0,"0.01")
        mesh_entry1.grid(row=1,column=1,sticky="E")

        mesh_label2 = ttk.Label(meshing_tab,text="Surface Mesh Flag")
        mesh_label2.grid(row=2,column=0,sticky="W")

        mesh_entry2 = tk.StringVar()
        mesh_entry2.set("True")
        mesh_entry2_options = ["True","False"]
        mesh_entry2_drop = tk.OptionMenu(meshing_tab,mesh_entry2,*mesh_entry2_options)
        mesh_entry2_drop.grid(row=2,column=1,sticky="E")

        mesh_label3 = ttk.Label(meshing_tab,text="Volume Mesh Flag")
        mesh_label3.grid(row=3,column=0,sticky="W")

        mesh_entry3 = tk.StringVar()
        mesh_entry3.set("True")
        mesh_entry3_options = ["True","False"]
        mesh_entry3_drop = tk.OptionMenu(meshing_tab,mesh_entry3,*mesh_entry3_options)
        mesh_entry3_drop.grid(row=3,column=1,sticky="E")

        mesh_label4 = ttk.Label(meshing_tab,text="No Merge")
        mesh_label4.grid(row=4,column=0,sticky="W")

        mesh_entry4 = tk.StringVar()
        mesh_entry4.set("False")
        mesh_entry4_options = ["True","False"]
        mesh_entry4_drop = tk.OptionMenu(meshing_tab,mesh_entry4,*mesh_entry4_options)
        mesh_entry4_drop.grid(row=4,column=1,sticky="E")

        mesh_label5 = ttk.Label(meshing_tab,text="Optimization Level")
        mesh_label5.grid(row=5,column=0,sticky="W")

        mesh_entry5 = tk.StringVar()
        mesh_entry5.set("5")
        mesh_entry5_options = ["1","2","3","4","5"]
        mesh_entry5_drop = tk.OptionMenu(meshing_tab,mesh_entry5,*mesh_entry5_options)
        mesh_entry5_drop.grid(row=5,column=1,sticky="E")

        mesh_label6 = ttk.Label(meshing_tab,text="Global Edge Size")
        mesh_label6.grid(row=6,column=0,sticky="W")

        mesh_entry6 = ttk.Entry(meshing_tab)
        mesh_entry6.insert(0,"18.0")
        mesh_entry6.grid(row=6,column=1,sticky="E")

        #Options for Simulation Tab
        sim_label1 = ttk.Label(simulation_tab,text="SV Presolver Filename")
        sim_label1.grid(row=1,column=0,sticky="W")

        sim_entry1 = ttk.Entry(simulation_tab)
        sim_entry1.insert(0,"cco")
        sim_entry1.grid(row=1,column=1,sticky="E")

        sim_label2 = ttk.Label(simulation_tab,text="Fluid Density")
        sim_label2.grid(row=2,column=0,sticky="W")

        sim_entry2 = ttk.Entry(simulation_tab)
        sim_entry2.insert(0,"1.06")
        sim_entry2.grid(row=2,column=1,sticky="E")

        sim_label3 = ttk.Label(simulation_tab,text="Fluid Viscosity")
        sim_label3.grid(row=3,column=0,sticky="W")

        sim_entry3 = ttk.Entry(simulation_tab)
        sim_entry3.insert(0,"0.04")
        sim_entry3.grid(row=3,column=1,sticky="E")

        sim_label4 = ttk.Label(simulation_tab,text="Initial Pressure")
        sim_label4.grid(row=4,column=0,sticky="W")

        sim_entry4 = ttk.Entry(simulation_tab)
        sim_entry4.insert(0,"0")
        sim_entry4.grid(row=4,column=1,sticky="E")

        sim_label5 = ttk.Label(simulation_tab,text="Initial Velocity")
        sim_label5.grid(row=5,column=0,sticky="W")

        sim_entry5 = ttk.Entry(simulation_tab)
        sim_entry5.insert(0,"[0.0001,0.0001,0.0001]")
        sim_entry5.grid(row=5,column=1,sticky="E")

        sim_label6 = ttk.Label(simulation_tab,text="BCT Analytical Shape")
        sim_label6.grid(row=6,column=0,sticky="W")

        sim_entry6 = ttk.Entry(simulation_tab)
        sim_entry6.insert(0,"parabolic")
        sim_entry6.grid(row=6,column=1,sticky="E")

        sim_label7 = ttk.Label(simulation_tab,text="Inflow Filename")
        sim_label7.grid(row=7,column=0,sticky="W")

        sim_entry7 = ttk.Entry(simulation_tab)
        sim_entry7.insert(0,"inflow")
        sim_entry7.grid(row=7,column=1,sticky="E")

        sim_label8 = ttk.Label(simulation_tab,text="Period")
        sim_label8.grid(row=8,column=0,sticky="W")

        sim_entry8 = ttk.Entry(simulation_tab)
        sim_entry8.insert(0,"1")
        sim_entry8.grid(row=8,column=1,sticky="E")

        sim_label9 = ttk.Label(simulation_tab,text="BCT Point Number")
        sim_label9.grid(row=9,column=0,sticky="W")

        sim_entry9 = ttk.Entry(simulation_tab)
        sim_entry9.insert(0,"2")
        sim_entry9.grid(row=9,column=1,sticky="E")

        sim_label10 = ttk.Label(simulation_tab,text="Fourier Mode")
        sim_label10.grid(row=10,column=0,sticky="W")

        sim_entry10 = ttk.Entry(simulation_tab)
        sim_entry10.insert(0,"1")
        sim_entry10.grid(row=10,column=1,sticky="E")

        sim_label11 = ttk.Label(simulation_tab,text="Pressures")
        sim_label11.grid(row=11,column=0,sticky="W")

        sim_entry11 = ttk.Entry(simulation_tab)
        sim_entry11.insert(0,"0")
        sim_entry11.grid(row=11,column=1,sticky="E")

        sim_label12 = ttk.Label(simulation_tab,text="svsolver Filename")
        sim_label12.grid(row=12,column=0,sticky="W")

        sim_entry12 = ttk.Entry(simulation_tab)
        sim_entry12.insert(0,"solver")
        sim_entry12.grid(row=12,column=1,sticky="E")

        sim_label13 = ttk.Label(simulation_tab,text="Number of Timesteps")
        sim_label13.grid(row=13,column=0,sticky="W")

        sim_entry13 = ttk.Entry(simulation_tab)
        sim_entry13.insert(0,"200")
        sim_entry13.grid(row=13,column=1,sticky="E")

        sim_label14 = ttk.Label(simulation_tab,text="Timestep Size")
        sim_label14.grid(row=14,column=0,sticky="W")

        sim_entry14 = ttk.Entry(simulation_tab)
        sim_entry14.insert(0,"0.02")
        sim_entry14.grid(row=14,column=1,sticky="E")

        sim_label15 = ttk.Label(simulation_tab,text="Number of Timesteps Between Restarts")
        sim_label15.grid(row=15,column=0,sticky="W")

        sim_entry15 = ttk.Entry(simulation_tab)
        sim_entry15.insert(0,"50")
        sim_entry15.grid(row=15,column=1,sticky="E")

        sim_label16 = ttk.Label(simulation_tab,text="Number of Force Surfaces")
        sim_label16.grid(row=16,column=0,sticky="W")

        sim_entry16 = ttk.Entry(simulation_tab)
        sim_entry16.insert(0,"1")
        sim_entry16.grid(row=16,column=1,sticky="E")

        sim_label17 = ttk.Label(simulation_tab,text="ID of Force Surface Calculation")
        sim_label17.grid(row=17,column=0,sticky="W")

        sim_entry17 = ttk.Entry(simulation_tab)
        sim_entry17.insert(0,"1")
        sim_entry17.grid(row=17,column=1,sticky="E")

        sim_label18 = ttk.Label(simulation_tab,text="Force Calculation Method")
        sim_label18.grid(row=18,column=0,sticky="W")

        sim_entry18 = ttk.Entry(simulation_tab)
        sim_entry18.insert(0,"Velocity Based")
        sim_entry18.grid(row=18,column=1,sticky="E")
        """
                          svpre_name='cco',fluid_density=1.06,
                          fluid_viscosity=0.04, initial_pressure=0,
                          initial_velocity=[0.0001,0.0001,0.0001],
                          bct_analytical_shape='parabolic',
                          inflow_file='inflow',
                          period=1,bct_point_number=2,
                          fourier_mode=1,pressures=0,
                          svsolver='solver',number_timesteps=200,
                          timestep_size=0.02,number_restarts=50,
                          number_force_surfaces=1,
                          surface_id_force_calc=1,
                          force_calc_method='Velocity Based',
                          print_avg_solution=True,
                          print_error_indicators=False,
                          varying_time_from_file=True,
                          step_construction='0 1 0 1',
                          pressure_coupling='Implicit',
                          backflow_stabilization=0.2,
                          residual_control=True,
                          residual_criteria=0.01,
                          minimum_req_iter=2,
                          svLS_type='NS',num_krylov=100,
                          num_solves_per_left=1,
                          tolerance_momentum=0.05,
                          tolerance_continuity=0.4,
                          tolerance_svLS_NS=0.4,
                          max_iter_NS=2,max_iter_momentum=4,
                          max_iter_continuity=400,
                          time_integration_rule='Second Order',
                          time_integration_rho=0.5,
                          flow_advection_form='Convective',
                          quadrature_interior=2,
                          quadrature_boundary=3,procs=24,
                          svpost='cco',start=0,vtu=True,
                          vtp=False,vtkcombo=False,
                          all_arg=True,wss=False,
                          sim_units_mm=False,
                          sim_units_cm=True,
                          global_edge_size=0.01,
                          distal_resistance=0
    """
