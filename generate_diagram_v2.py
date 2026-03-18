import base64
import zlib
import urllib.request
import os
import json

def generate_png(mermaid_code, output_path):
    try:
        compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
        encoded_str = base64.urlsafe_b64encode(compressed).decode('utf-8')
        url = f"https://kroki.io/mermaid/png/{encoded_str}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        print(f"Downloading from Kroki: {url}")
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully saved to {output_path}")
    except Exception as e:
        print(f"Kroki Error for {output_path}: {e}")

# 1. Main Pipeline Diagram (Clean & High-Level)
main_pipeline_mermaid = """
graph LR
    A[Data Profiling] --> B[Data Preprocessing]
    B --> C[Feature Power Ranking]
    C --> D{Model Build}
    D --> E[Validation Suite]
    E --> F[Report Generation]

    classDef preproc fill:#e8f4f8,stroke:#2b7b9c,stroke-width:2px,font-size:36px,padding:20px,color:#1a4a5e;
    classDef modeling fill:#f3e8f8,stroke:#6b2b9c,stroke-width:2px,font-size:36px,padding:20px,color:#401a5e;
    classDef validation fill:#e8f8ec,stroke:#2b9c4c,stroke-width:2px,font-size:36px,padding:20px,color:#1a5e2e;
    classDef reporting fill:#fff3e0,stroke:#e65100,stroke-width:2px,font-size:36px,padding:20px,color:#4d1b00;

    class A,B preproc;
    class C,D modeling;
    class E validation;
    class F reporting;
"""

# 2. Validation Suite Detail Diagram (Independent Modules, NO ARROWS)
validation_detail_mermaid = """
graph LR
    subgraph VS ["Validation Suite Components"]
        direction LR
        V1["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Explainability</div>"]
        V2["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Stress Testing</div>"]
        V3["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Benchmarking</div>"]
        V4["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Cluster Coverage</div>"]
        V5["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Drift Analysis</div>"]
        V6["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Diagnostic Plots</div>"]
        V7["<div style='width:250px; height:100px; display:flex; align-items:center; justify-content:center;'>Metrics Comparison</div>"]
        
        %% Use invisible links to force horizontal alignment without showing arrows or lines
        V1 ~~~ V2 ~~~ V3 ~~~ V4 ~~~ V5 ~~~ V6 ~~~ V7
    end

    %% High-Quality Styling (Synced with first diagram)
    classDef validation fill:#e8f8ec,stroke:#2b9c4c,stroke-width:2px,font-size:32px,padding:15px,color:#1a5e2e;
    classDef suite fill:#f9fff9,stroke:#2b9c4c,stroke-width:4px,font-size:42px,padding:30px,color:#1a5e2e;

    class V1,V2,V3,V4,V5,V6,V7 validation;
    class VS suite;
"""

generate_png(main_pipeline_mermaid, "/Users/dolly/TanML v11/architecture_main.png")
generate_png(validation_detail_mermaid, "/Users/dolly/TanML v11/architecture_validation.png")
