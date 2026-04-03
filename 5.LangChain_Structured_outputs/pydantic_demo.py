from pydantic import BaseModel, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str = 'amrit'
    age: Optional[int] = None
    email: str

new_student = {'name':'raj', 'age':'21', 'email':'abc@gmail.com'}

student = Student(**new_student)

print(student)