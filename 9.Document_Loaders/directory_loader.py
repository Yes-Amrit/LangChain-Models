from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import sys
import io
# Ensure stdout uses UTF-8 encoding in a way that's recognized by type checkers
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader,  # type: ignore[arg-type]
)

docs=loader.load()

print(docs[15].page_content)
print(docs[430].metadata)