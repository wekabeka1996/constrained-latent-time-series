# tools/phase2/run_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py

import sys
import json

# Crucial: No direct torch import at top level or anywhere in this file.

def main():
    if len(sys.argv) > 1:
        print("Error: run_p55_fc_vae_encoder_posterior_kl_boundary_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.fc_vae_encoder_posterior_kl_boundary import (
            run_fc_vae_encoder_posterior_kl_probe,
            fc_vae_encoder_posterior_kl_probe_to_json_dict,
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P55",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_fc_vae_encoder_posterior_kl_probe()
    final_dict = fc_vae_encoder_posterior_kl_probe_to_json_dict(probe_res)
    
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict.get("verdict") == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
