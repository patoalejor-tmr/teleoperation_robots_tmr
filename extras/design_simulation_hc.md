### Steps for simulation
The following steps are suggested for the simulation of the model inference

- Step 1. Collect data `A: <user>_<task>_<trial>.<format>`, maybe using parquet or h5py  
- Step 2. Use data `A` for training a model `M`  
- Step 3. Use any subset from `A` for testing model `M` and save the inference into `B`  
    - The output from `M` should be something like `(Ts, Ac, DoF)` were      
    - **Ts**: timesteps recorded at fixed FPS from camera   
    - **Ac**: Action Chunk size from VLA model (eg. 16)  
    - **DoF**: Robot degree of freedom (16-19-59) for RB-Y1 or AIW with 2G or 5G  
    - *Note: if onely one action is predicted then `Ac=1`*  
- Step 4. Use inference data from `B` to simulate the robot `R`
    - It would be convenient to save the data such as `<model>_<task>_<robot>.<format>`
    - If the format to save is decided beforehand also is helpful (eg., .npy, .paquet, .pickle)

### Image reference for HC

For steps 1 to 3 can be done in the following screen add this functionality from `Edge-Team`

![alt text](image-2.png)

Then to enable simulation select option in the general screen `Console-team`

![alt text](image.png)

Finally select the file of inference data that you want to see in simulation and play the button

![alt text](image-1.png)
