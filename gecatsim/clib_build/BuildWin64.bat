ECHO OFF
REM Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license
ECHO ON

set PATH=C:\PROGRA~1\mingw64\bin;%PATH%
CD src

REM Step 1: clean first
C:\PROGRA~1\mingw64\bin\mingw32-make -f ..\MakeWindows64 clean

REM Step 2: rebuild
C:\PROGRA~1\mingw64\bin\mingw32-make -f ..\MakeWindows64

REM Step 3: move DLL
move /Y libcatsim64.dll ..\..\lib

@PAUSE

CD ..

@ECHO Windows build complete
@PAUSE