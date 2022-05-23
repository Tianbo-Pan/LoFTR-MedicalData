# LoFTR-MedicalData
Compare to the author's original work： [LoFTR](https://github.com/zju3dv/LoFTR#readme)  
Use the 'create_homo_data/Process_data.py' to process each frame to get the folder 'xx_processed'  
### Set the soft links in data/homo/test by using:  
```shell
ln -s train_processed/* data/homo/test/
ln -s val_processed/* data/homo_val/test/
```
After adding soft links:
![demo_vid](assets/soft-links.PNG)  

### Make the intrinsics.npz  
- Using the 'get_intrinsics.py' to make the intrinsics.npz and homo_finetune.npz(they are same)  
- put the intinsics.npz and homo_finetune.npz into assets/homo_fientune  
- the content of homo_list_path.txt is the filename of homo_finetune.npz  
:triangular_flag_on_post: val_homo is also treated in the same way. Shown below:
![intrinsic](assets/val_intrinsic.PNG)
