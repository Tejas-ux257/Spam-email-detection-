
import csv
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def analyze_data(file_path):
    data = []
    total_sales = 0
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
           
            row['Sales'] = int(row['Sales'])
            total_sales += row['Sales']
            data.append(row)
    average_sales = total_sales / len(data)
    return data, total_sales, average_sales

def generate_chart(data):
    names = [row['Name'] for row in data]
    sales = [row['Sales'] for row in data]

    plt.figure(figsize=(8, 4))
    plt.bar(names, sales, color='skyblue')
    plt.title('Employee Sales Report')
    plt.xlabel('Employee Name')
    plt.ylabel('Sales Amount')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("sales_chart.png")
    plt.close()

def generate_pdf_report(data, total_sales, average_sales, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Automated Sales Report", styles['Title']))
    elements.append(Spacer(1, 12))

    summary = f"<b>Total Sales:</b> ₹{total_sales}<br/><b>Average Sales:</b> ₹{round(average_sales, 2)}"
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Sales Performance Chart</b>", styles['Heading2']))
    elements.append(Image("sales_chart.png", width=400, height=200))
    elements.append(Spacer(1, 12))
    
    table_data = [["EmployeeID", "Name", "Department", "Sales", "Q1", "Q2", "Q3", "Q4"]]
    for row in data:
        table_data.append([row['EmployeeID'], row['Name'], row['Department'], row['Sales'], row['Q1'], row['Q2'], row['Q3'], row['Q4']])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Report generated automatically using Python 🐍", styles['Italic']))

    doc.build(elements)

if __name__ == "__main__":
    data_file = r"C:\Users\Manoj Vadhi\OneDrive\Desktop\automated pdf generater t 2\data.csv"  # Your data.csv path
    output_pdf = "sample_report_v2.pdf"


    data, total, average = analyze_data(data_file)
    generate_chart(data)
    generate_pdf_report(data, total, average, output_pdf)

    print(f" ✅ Report successfully created: {output_pdf}")

    