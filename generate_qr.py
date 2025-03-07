import qrcode

deployed_url = "https://crime-analysis.onrender.com"
qr = qrcode.make(deployed_url)
qr.save("crime_analysis_qr.png")

print("QR code saved as 'crime_analysis_qr.png'.")
