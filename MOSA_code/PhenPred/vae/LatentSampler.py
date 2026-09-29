import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import gc


class LatentSampler:
    """
    Samples from the VAE latent space distribution after training.
    """
    
    def __init__(self, hypers, save_dir="reports/vae/files"):
        """
        Initialize the latent sampler.
        
        Args:
            hypers: Hyperparameters dictionary
            save_dir: Directory to save the sampled latent cube
        """
        self.hypers = hypers
        self.save_dir = save_dir
        self.should_sample = hypers.get("sample_latent", False)
        self.num_samples = hypers.get('cube_sample_size', 100)
        self.epoch_sampling = hypers.get("epoch_sampling", False)
        print(f"LatentSampler initialized. epoch_sampling={self.epoch_sampling}")
        self.variance_history = []
        
    def sample_latent(self, model, dataloader, device="cuda"):
        """
        Sample from the latent space distribution.
        
        Args:
            model: Trained VAE model
            dataloader: DataLoader containing the training data
            device: Device to run on
            
        Returns:
            Sampled latent representations as numpy array
        """
        if not self.should_sample:
            print("Latent sampling is disabled. Set 'sample_latent': true in hyperparameters.json")
            return None
            
        model.eval()
        all_samples = []
        
        with torch.no_grad():
            # Get mu and log_var from the entire dataset
            for batch in dataloader:
                # Unpack batch data
                x, y, _, x_mask = batch
                print(batch)
                print(f"Sampling latent for batch size: {x[0].shape[0]}")
                # Apply masking to features
                x_masked = [m[:, x_mask[i][0]] for i, m in enumerate(x)]
                
                # Prepare input for model based on use_conditionals setting
                if self.hypers.get("use_conditionals", False):
                    model_input = x_masked + [y]
                else:
                    model_input = x_masked
                
                # Forward pass to get mu and log_var
                out = model(model_input)
                mu = out["mu"]  # Extract mu from output dictionary
                log_var = out["log_var"]  # Extract log_var from output dictionary
                
                # Sample from the distribution self.num_samples times
                batch_samples = []
                for _ in range(self.num_samples):
                    # Reparameterization trick: z = mu + sigma * epsilon
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z_sample = mu + eps * std
                    batch_samples.append(z_sample.cpu().numpy())
                
                # Stack samples: [num_samples, batch_size, latent_dim]
                batch_samples = np.stack(batch_samples, axis=0)
                all_samples.append(batch_samples)
        
        # Concatenate all batches: [num_samples, total_samples, latent_dim]
        all_samples = np.concatenate(all_samples, axis=1)
        
        # Save the sampled latent cube
        self._save_latent_cube(all_samples)
        
        return all_samples
    
    def _save_latent_cube(self, latent_cube):
        """
        Save the latent cube to disk.
        
        Args:
            latent_cube: Numpy array of shape [num_samples, total_samples, latent_dim]
        """
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{timestamp}_latent_joint_cube.npy"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save the cube
        np.save(filepath, latent_cube)
        
        print(f"Latent cube saved to: {filepath}")
        print(f"Shape: {latent_cube.shape}")
        print(f"[num_samples={latent_cube.shape[0]}, total_data_points={latent_cube.shape[1]}, latent_dim={latent_cube.shape[2]}]")
        
        return filepath

    def generate_decoded_cubes(self, model, train, timestamp):
        """
        Generate decoded samples for all views by passing latent samples through decoders.
        This creates a cube of generated samples for each view.
        
        Args:
            model: Trained VAE model
            train: CLinesTrain object containing data and device information
            timestamp: Timestamp string for file naming
        """
        if not self.hypers.get("sample_cube", False):
            print("Cube sampling is disabled. Set 'sample_cube': true in hyperparameters.json")
            return
        
        print("\n" + "="*50)
        print("Generating decoded cubes for all views")
        print("="*50)
        
        model.eval()
        generated_samples = {}
        
        with torch.no_grad():
            # Process each view
            for view_name in train.data.view_names:
                print(f"\nProcessing view: {view_name}")
                samples = []
                
                # Get all data in one batch
                from torch.utils.data import DataLoader
                data_all = DataLoader(
                    train.data, batch_size=len(train.data.samples), shuffle=False
                )
                
                for data in data_all:
                    x, y, _, x_mask = data
                    
                    # Apply masking to features
                    x_masked = [m[:, x_mask[i][0]] for i, m in enumerate(x)]
                    
                    # Prepare input for model
                    if self.hypers.get("use_conditionals", False):
                        model_input = x_masked + [y]
                    else:
                        model_input = x_masked
                    
                    # Forward pass to get mu and log_var
                    out = model(model_input)
                    mu = out["mu"]
                    log_var = out["log_var"]
                    
                    # Generate multiple samples from the latent distribution
                    for sample_idx in range(self.num_samples):
                        # Reparameterization trick: z = mu + sigma * epsilon
                        std = torch.exp(0.5 * log_var)
                        eps = torch.randn_like(std)
                        latent_sample = mu + eps * std
                        
                        # Decode the latent sample
                        if self.hypers.get("use_conditionals", False):
                            # Concatenate latent sample with conditional inputs
                            conditional_inputs = y.to(train.device).float()
                            latent_with_conditionals = torch.cat([latent_sample, conditional_inputs], dim=1)
                            decoded_samples = model.module.decoders[train.data.view_names.index(view_name)](latent_with_conditionals)
                        else:
                            decoded_samples = model.module.decoders[train.data.view_names.index(view_name)](latent_sample)
                        
                        # Un-standardize the outputs
                        decoded_samples = train.data.view_scalers[view_name].inverse_transform(decoded_samples.cpu().numpy())

                        # Post-processing
                        if view_name == "copynumber":
                            decoded_samples = np.round(decoded_samples)
                            decoded_samples = np.clip(decoded_samples, -2, 2)
                        else:
                            decoded_samples = np.round(decoded_samples, 5)

                        # Convert to DataFrame with proper labels
                        decoded_samples_df = pd.DataFrame(
                            decoded_samples,
                            index=train.data.samples,
                            columns=train.data.view_feature_names[view_name]
                        )
                        
                        samples.append(decoded_samples_df)
                
                # Stack samples along the third dimension to create a cube
                generated_samples[view_name] = np.stack([df.values for df in samples], axis=-1)

                # Save the cube to a file
                output_path = f"{self.save_dir}/{timestamp}_generated_{view_name}_cube.npy"
                np.save(output_path, generated_samples[view_name])
                print(f"Saved generated samples for {view_name} to {output_path}")
                print(f"  Shape: {generated_samples[view_name].shape}")
                print(f"  [samples={generated_samples[view_name].shape[0]}, features={generated_samples[view_name].shape[1]}, replications={generated_samples[view_name].shape[2]}]")

                # Clean up memory/cache for previous cube
                del samples
                del generated_samples[view_name]
                gc.collect()
        
        print("="*50)
        print("Finished generating decoded cubes")
        print("="*50 + "\n")

    def sample_epoch_variance(self, model, dataset, device="cuda"):
        """
        Sample the variance of the latent space for the current epoch.
        
        Args:
            model: Trained VAE model
            dataset: Dataset containing the training data
            device: Device to run on
        """
        if not self.epoch_sampling:
            # print("Skipping epoch variance sampling (disabled)")
            return

        # print("Sampling epoch variance...")
        model.eval()
        epoch_variances = []
        
        # Create a sequential dataloader to ensure consistent sample order
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.hypers["batch_size"], 
            shuffle=False, 
            drop_last=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                x, y, _, x_mask = batch
                
                x = [m.to(device) for m in x]
                x_masked = [m[:, x_mask[i][0]].to(device) for i, m in enumerate(x)]
                y = y.to(device)
                
                if self.hypers.get("use_conditionals", False):
                    model_input = x_masked + [y]
                else:
                    model_input = x_masked
                
                out = model(model_input)
                log_var = out["log_var"]
                
                # Convert log_var to variance
                variance = torch.exp(log_var)
                epoch_variances.append(variance.cpu().numpy())
        
        # Concatenate batches: [total_samples, latent_dim]
        epoch_variances = np.concatenate(epoch_variances, axis=0)
        self.variance_history.append(epoch_variances)
        # print(f"Captured variance for epoch. History size: {len(self.variance_history)}")

    def save_variance_cube(self, timestamp):
        """
        Save the variance history as a cube.
        
        Args:
            timestamp: Timestamp string for file naming
        """
        print(f"Attempting to save variance cube. History length: {len(self.variance_history)}")
        if not self.variance_history:
            print("Variance history is empty. Nothing to save.")
            return

        # Stack history: [total_samples, latent_dim, num_epochs]
        variance_cube = np.stack(self.variance_history, axis=-1)
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        filename = f"{timestamp}_variance_cube.npy"
        filepath = os.path.join(self.save_dir, filename)
        
        np.save(filepath, variance_cube)
        
        print(f"Variance cube saved to: {filepath}")
        print(f"Shape: {variance_cube.shape}")
        
        # Clear history to free memory
        self.variance_history = []
