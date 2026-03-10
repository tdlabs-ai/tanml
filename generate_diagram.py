import base64
import zlib
import urllib.request
import os
import json

mermaid_code = """
graph LR
    A[Data Profiling] --> B[Data Preprocessing]
    B --> C[Feature Power Ranking]
    C --> D{Model Build}
    D --> H[Validation Suite]
    
    %% HTML-like labels to enforce exact same width/height for all 7 boxes
    H --> V1["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Metrics Comparison</div>"]
    H --> V2["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Diagnostic Plots</div>"]
    H --> V3["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Drift Analysis</div>"]
    H --> V4["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Cluster Coverage</div>"]
    H --> V5["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Benchmarking</div>"]
    H --> V6["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Stress Testing</div>"]
    H --> V7["<div style='width:300px; height:100px; display:flex; align-items:center; justify-content:center;'>Explainability</div>"]
    
    V1 --> L[Report Generation]
    V2 --> L
    V3 --> L
    V4 --> L
    V5 --> L
    V6 --> L
    V7 --> L

    %% Styling for LARGE fonts, uniform boxes, and COLORS
    classDef preproc fill:#e8f4f8,stroke:#2b7b9c,stroke-width:2px,font-size:36px,padding:20px,color:#1a4a5e;
    classDef modeling fill:#f3e8f8,stroke:#6b2b9c,stroke-width:2px,font-size:36px,padding:20px,color:#401a5e;
    classDef validation fill:#e8f8ec,stroke:#2b9c4c,stroke-width:2px,font-size:36px,padding:20px,color:#1a5e2e;
    classDef reporting fill:#fff3e0,stroke:#e65100,stroke-width:2px,font-size:36px,padding:20px,color:#4d1b00;
    
    class A,B preproc;
    class C,D modeling;
    class H,V1,V2,V3,V4,V5,V6,V7 validation;
    class L reporting;
"""

output_path = "/Users/dolly/TanML v11/architecture.png"

try:
    compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
    encoded_str = base64.urlsafe_b64encode(compressed).decode('utf-8')
    url = f"https://kroki.io/mermaid/png/{encoded_str}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print(f"Downloading from Kroki: {url}")
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        out_file.write(response.read())
    
    print(f"Successfully saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
except Exception as e:
    print(f"Kroki Error: {e}")
    # Fallback to mermaid.ink
    try:
        j = {
          "code": mermaid_code.strip(),
          "mermaid": {
            "theme": "default",
          }
        }
        encoded_str = base64.urlsafe_b64encode(json.dumps(j).encode('utf-8')).decode('utf-8').rstrip("=")
        url2 = f"https://mermaid.ink/img/{encoded_str}?bgColor=ffffff"
        req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        print(f"Downloading from Mermaid.ink: {url2}")
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully saved from mermaid.ink to {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    except Exception as e2:
        print(f"Mermaid.ink Error: {e2}")
