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
                    body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f0f2f5; color: #333; }}
                    .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }}
                    h1 {{ color: #1a1a1a; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }}
                    h2 {{ color: #2c3e50; margin-top: 0; }}
                    h3 {{ color: #34495e; margin-bottom: 10px; }}
                    
                    /* Table Styling */
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                    th {{ background-color: #f8f9fa; font-weight: 600; color: #555; }}
                    tr:last-child td {{ border-bottom: none; }}
                    tr:hover {{ background-color: #fcfcfc; }}
                    
                    .metric-name {{ color: #555; font-weight: 500; }}
                    .metric-value {{ color: #222; font-weight: 700; }}
                    .metric-pass {{ background-color: #d4edda; color: #155724; }}
                    .metric-warn {{ background-color: #fff3cd; color: #856404; }}
                    .metric-fail {{ background-color: #f8d7da; color: #721c24; }}
                    
                    img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #eee; }}
                    .plot-container {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>QC Report: {subject}</h1>
                <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Summary Metrics</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f'<tr class="{self._get_row_class(k, v)}"><td class="metric-name">{k}</td><td class="metric-value">{v}</td></tr>' for k, v in metrics.items()])}
                        </tbody>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Visualizations</h2>
                    {''.join([f'<div class="plot-container"><h3>{k}</h3><img src="{v}" alt="{k}"/></div>' for k, v in plots.items()])}
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

    def _get_row_class(self, key, value):
        if key == "QC Decision":
            if value == "PASS": return "metric-pass"
            if value == "WARN": return "metric-warn"
            if value == "FAIL": return "metric-fail"
        return ""
