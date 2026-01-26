import os
import datetime

class QCReport:
    """
    Generates an HTML QC report.
    """
    def generate(self, subject: str, output_path: str, metrics: dict, plots: dict) -> bool:
        """
        metrics: dict of scalar values (e.g. mean FD)
        plots: dict of paths to plot images
        """
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QC Report: {subject}</title>
                <style>
                    body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }}
                    .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                    h1 {{ color: #1a1a1a; }}
                    .metric {{ display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 10px 0; }}
                    .metric:last-child {{ border-bottom: none; }}
                    img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <h1>QC Report: {subject}</h1>
                <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Summary Metrics</h2>
                    {''.join([f'<div class="metric"><span>{k}</span><strong>{v}</strong></div>' for k, v in metrics.items()])}
                </div>
                
                <div class="card">
                    <h2>Visualizations</h2>
                    {''.join([f'<h3>{k}</h3><img src="{v}" alt="{k}"/>' for k, v in plots.items()])}
                </div>
            </body>
            </html>
            """
            
            with open(output_path, "w") as f:
                f.write(html)
            return True
        except Exception as e:
            print(f"QC report generation failed: {e}")
            return False
