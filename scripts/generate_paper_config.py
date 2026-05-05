import json
import sys
import os

# Add the parent directory to sys.path so we can import compulsionbench
sys.path.append(os.path.abspath("compulsionbench"))

try:
    from compulsionbench import BenchConfig
except ImportError:
    # Try alternate import if structure is different
    from compulsionbench.compulsionbench import BenchConfig

def main():
    # Instantiate the default config (Paper Defaults)
    cfg = BenchConfig()
    
    # Save to JSON
    output_path = "config_calibrated.json"
    cfg.save(output_path)
    print(f"✅ Created {output_path} with Paper Defaults.")
    
    # Verify content
    with open(output_path, "r") as f:
        data = json.load(f)
        print(f"   Sample param: rho_h_loc = {data.get('rho_h_loc')}")

if __name__ == "__main__":
    main()
