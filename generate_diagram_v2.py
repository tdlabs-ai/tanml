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

# Unified Architecture Diagram
unified_architecture_mermaid = """
graph LR
    A[Data Profiling] --> B[Data Preprocessing]
    B --> C[Feature Power Ranking]
    C --> MB
    MB --> VS
    VS --> F[Report Generation]

    subgraph MB [Model Build]
        direction TB
        D1[Regression]
        D2[Classification]
    end

    subgraph VS [Validation Suite]
        direction TB
        V1[Explainability]
        V2[Stress Testing]
        V3[Benchmarking]
        V4[Cluster Coverage]
        V5[Drift Analysis]
        V6[Diagnostic Plots]
        V7[Metrics Comparison]
    end

    %% Styling
    classDef blue fill:#e8f4f8,stroke:#2b7b9c,stroke-width:2px,font-size:30px,padding:15px,color:#1a4a5e;
    classDef purple fill:#f3e8f8,stroke:#6b2b9c,stroke-width:2px,font-size:30px,padding:15px,color:#401a5e;
    classDef green fill:#e8f8ec,stroke:#2b9c4c,stroke-width:2px,font-size:30px,padding:15px,color:#1a5e2e;
    classDef orange fill:#fff3e0,stroke:#e65100,stroke-width:2px,font-size:30px,padding:15px,color:#4d1b00;
    
    classDef mb_cont fill:#f3e8f8,stroke:#6b2b9c,stroke-width:3px,font-size:34px,color:#401a5e;
    classDef vs_cont fill:#e8f8ec,stroke:#2b9c4c,stroke-width:3px,font-size:34px,color:#1a5e2e;

    class A,B blue;
    class C,D1,D2 purple;
    class V1,V2,V3,V4,V5,V6,V7 green;
    class F orange;
    
    class MB mb_cont;
    class VS vs_cont;
"""

generate_png(unified_architecture_mermaid, "/Users/dolly/TanML v11/architecture.png")
