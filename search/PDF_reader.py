import pdfplumber

def read_pdf(pdf_path):
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text#+ "\n"
    
    return full_text

# 使用示例
if __name__ == "__main__":
    pdf_path = "data.pdf"  # 你的PDF文件名
    
    text = read_pdf(pdf_path)
    
    print(f"提取了 {len(text)} 个字符")
    print("\n前500个字符:")
    print(text[:500])
    
    # 保存提取的文本
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n文本已保存到: extracted_text.txt")