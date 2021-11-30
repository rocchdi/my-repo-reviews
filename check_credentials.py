from fastapi import Request
from fastapi import HTTPException





#
credentials_base={
  "alice": "wonderland",
  "bob": "builder",
  "clementine": "mandarine",
  "admin": "4dm1N"
}

Admin_user="admin"
Admin_password="4dm1N"



#
#check_credentials
#
def check_credentials(request,profile):
    Authorized=False

    headers = request.headers
    if "authorization-header" not in headers :
        raise HTTPException(status_code=400, detail="invalid authorization")
    else:
        authorization=headers['authorization-header']
        authorization_list = list(authorization.split(" "))
        if len(authorization_list) !=2:
            raise HTTPException(status_code=400, detail="invalid authorization")
        else:
            if 'Basic' not in authorization_list:
                raise HTTPException(status_code=400, detail="invalid authorization")
            else:
                credentials=authorization_list[1]
                credentials = credentials.split(":")
                if len(credentials) != 2:
                    raise HTTPException(status_code=400, detail="invalid authorization")
                else:
                    user=str(credentials[0])
                    password=str(credentials[1])
                    if user not in credentials_base:
                        raise HTTPException(status_code=401,detail="Unauthorized access")
                    else:
                        if profile != "Administrator":
                            base_password=credentials_base[user]
                            if password!=base_password:
                                raise HTTPException(status_code=403,detail="Access Forbidden")
                            else:
                                Authorized=True
                        else:
                            if user != Admin_user or password != Admin_password:
                                raise HTTPException(status_code=403,detail="Access Forbidden")
                            else:
                                Authorized=True


    return Authorized

