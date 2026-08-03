import os
import PyPDF2
import docx

class FileParser:
    @staticmethod
    def extract_text(file_storage) -> tuple:
        """
        Extracts plain text from file_storage object (TXT, PDF, DOCX).
        Returns tuple: (extracted_text, filename, file_type)
        """
        filename = getattr(file_storage, 'filename', '') or 'uploaded_document.txt'
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        
        extracted_text = ""
        
        try:
            if ext == 'txt':
                extracted_text = file_storage.read().decode('utf-8', errors='ignore')
            elif ext == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file_storage)
                pages_text = []
                for page in pdf_reader.pages:
                    txt = page.extract_text()
                    if txt:
                        pages_text.append(txt)
                extracted_text = "\n\n".join(pages_text)
            elif ext == 'docx':
                doc = docx.Document(file_storage)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                extracted_text = "\n\n".join(paragraphs)
            else:
                raise ValueError(f"Unsupported file extension: .{ext}")
        except Exception as e:
            raise RuntimeError(f"Error parsing file '{filename}': {str(e)}")

        return extracted_text.strip(), filename, ext.upper()
