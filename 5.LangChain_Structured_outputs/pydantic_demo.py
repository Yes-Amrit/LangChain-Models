from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'amrit'
    age: Optional[int] = None
    email: str
    CGPA: float = Field(gt=0, lt=10.5, default = 9)

new_student = {'name':'raj', 'age':'21', 'email':'abc@gmail.com', 'CGPA':4}

student = Student(**new_student)

student_dict = student.model_dump()
student_json = student.model_dump_json()
print(student_dict['CGPA'])
print(student_json)