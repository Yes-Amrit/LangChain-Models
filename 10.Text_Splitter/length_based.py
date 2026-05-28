from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# text = """
# Space exploration refers to the investigation of outer space using astronomy, robotic probes, and human spaceflight. It began in earnest during the mid-20th century with milestones such as the launch of Sputnik 1 by the Soviet Union space program, which marked the start of the space age. This was followed by significant achievements from NASA, including the historic Apollo 11 Moon Landing. Over time, space exploration has expanded beyond national competition into international collaboration, exemplified by projects like the International Space Station, where astronauts from multiple countries conduct scientific research in microgravity. These efforts have greatly enhanced our understanding of planetary systems, cosmic phenomena, and the origins of the universe.

# In the modern era, space exploration is increasingly driven by both government agencies and private companies such as SpaceX and Blue Origin. The focus has shifted toward long-term sustainability in space, including missions to Mars, asteroid mining, and the development of reusable launch systems. Advanced telescopes like the James Webb Space Telescope are enabling scientists to observe distant galaxies and study the early universe with unprecedented clarity. Space exploration also plays a crucial role in technological innovation, contributing to advancements in communication, navigation, and materials science, while raising important questions about space ethics, planetary protection, and the future of human civilization beyond Earth.
# """

loader = PyPDFLoader('poem.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)

# result = splitter.split_text(text)
result = splitter.split_documents(docs)

print(result[0])