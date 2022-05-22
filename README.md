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
