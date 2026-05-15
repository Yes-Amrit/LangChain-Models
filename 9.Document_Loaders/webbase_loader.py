import os
os.environ["USER_AGENT"] = "Mozilla/5.0"
from langchain_community.document_loaders import WebBaseLoader



url = 'https://www.amazon.in/Portable-Cooler-Water-Tank-Power/dp/B0GYRVW25V/ref=asc_df_B0GYRVW25V?mcid=fd76bd9098343d8baf4d0a4722974115&tag=googleshopdes-21&linkCode=df0&hvadid=709856187270&hvpos=&hvnetw=g&hvrand=5785233848939153937&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9180276&hvtargid=pla-2483317149330&psc=1&hvocijid=5785233848939153937-B0GYRVW25V-&hvexpln=0&gad_source=1'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)