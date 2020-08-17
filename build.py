
import os
import os.path
import sys

common_sources = ["Block.cpp","Activation.cpp"]
#preprocessor_sources = []
solver_sources = ["Cascient.cpp"]
#postprocessor_sources = []

compiler = "g++"
include_arg = "-I../../dev/src/dps/drop_dynamics/ezdev/"
link_arg = "../../dev/src/dps/drop_dynamics/ezdev/libezdev.a -lpng"

#preprocessor_output = "pre_davinci"
solver_output = "cascient"
#postprocessor_output = "post_davinci"

object_extension = ".o"

print_only = False

def NeedToCompile(sourcefile,objectfile):
	# check if object file exists
	if(os.path.isfile(objectfile)):
		# check its time signature
		sourcetime = os.path.getmtime(sourcefile)
		objecttime = os.path.getmtime(objectfile)
		# leave a tolerance of 10 seconds to avoid any inaccurate time reporting
		if(sourcetime < (objecttime - 10)):
			return False
	return True

def Execute(command):
	if(print_only):
		print(command)
	else:
		print(command)
		os.system(command)

def GetObjectName(sourcename):
	objectname = sourcename[0:sourcename.rfind('.')] + object_extension
	return objectname

def CompileSource(sourcefile,debug):
	if(NeedToCompile(sourcefile,GetObjectName(sourcefile))):
		if(debug):	compile_string = compiler + " -Wall -Wextra -pedantic -ansi -g " + include_arg + " -c " + sourcefile
		else:		compile_string = compiler + " -Wall -Wextra -pedantic -ansi -O2 " + include_arg + " -c " + sourcefile
		Execute(compile_string)
		return True
	return False

def CompileSources(source_list,debug):
	for source in source_list:
		CompileSource(source,debug)

def LinkObjects(source_list,output):
	link_command = compiler + " "
	for source in source_list:
		link_command = link_command + GetObjectName(source) + " "
	link_command = link_command + " " + link_arg + " -o " + output
	Execute(link_command)

def Compile(debug):
	CompileSources(common_sources,debug)
	#CompileSources(preprocessor_sources)
	CompileSources(solver_sources,debug)
	#CompileSources(postprocessor_sources,debug)

def Link():
	#LinkObjects(common_sources + preprocessor_sources,preprocessor_output)
	LinkObjects(common_sources + solver_sources,solver_output)
	#LinkObjects(common_sources + postprocessor_sources,postprocessor_output)


clean = False
debug = False

for i in range(1,len(sys.argv)):
	if(sys.argv[i] == "clean"):
		clean = True
	elif(sys.argv[i] == "debug"):
		debug = True

if(clean):
	Execute("rm *.o")
	Execute("rm " + solver_output)
	#Execute("rm " + postprocessor_output)

Compile(debug)
Link()

