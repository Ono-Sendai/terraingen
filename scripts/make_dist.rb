#=====================================================================
#make_dist.rb
#------------
#Copyright Nicholas Chapman 2025 -
#=====================================================================

require 'FileUtils'
require './make_dist_private.rb'


def copyAllFilesInDir(dir_from, dir_to)
	if !Dir.exist?(dir_from)
		STDERR.puts "Error: Source directory doesn't exist: #{dir_from}"
		exit(1)
	end
	
	if !Dir.exist?(dir_to)
		STDERR.puts "Error: Target directory doesn't exist: #{dir_to}"
		exit(1)
	end

	Dir.foreach(dir_from) do |f|
		FileUtils.cp(dir_from + "/" + f, dir_to, verbose: true) if f != ".." && f != "."
	end
end

def getAndCheckEnvVar(name)
	env_var = ENV[name]

	if env_var.nil?
		STDERR.puts "#{name} env var not defined."
		exit(1)
	end

	puts "#{name}: #{env_var}"
	return env_var
end

def copyVCRedist(vs_version, target_dir)

	redist_path = "C:/Program Files/Microsoft Visual Studio/#{vs_version}/Community/VC/Redist/MSVC/14.42.34433/x64"
	copyAllFilesInDir("#{redist_path}/Microsoft.VC143.CRT", target_dir)

end

def exec_command(cmd)
	res = Kernel.system(cmd)
	if !res
		STDERR.puts "Command failed: " + cmd
		exit(1)
	end
end



dist_dir = "d:/files/temp_terraingen_dist"

FileUtils.rm_r(dist_dir, :verbose=>true) if File.exist?(dist_dir)
FileUtils.mkdir(dist_dir, :verbose=>true)

# Copy and sign executable and SDL DLL
FileUtils.cp("c:\\programming\\terraingen\\vs2022_build\\Release\\terraingen.exe", dist_dir, :verbose=>true)
sign_app("TerrainGen", dist_dir + "/terraingen.exe")

FileUtils.cp("C:\\programming\\SDL\\sdl_2.30.0_build\\Release\\SDL2.dll", dist_dir, :verbose=>true)
sign_app("SDL", dist_dir + "/SDL2.dll")

# Copy VC redist
copyVCRedist(2022, dist_dir)

# Copy resources
FileUtils.cp_r("../resources", dist_dir, :verbose => true)

# Copy erosion_kernel.cl
FileUtils.cp_r("../erosion_kernel.cl", dist_dir, :verbose => true)


# Copy OpenGL shaders and OpenGL data.
glare_core_repos_dir = getAndCheckEnvVar('GLARE_CORE_TRUNK_DIR')
FileUtils.cp_r("#{glare_core_repos_dir}/opengl/shaders", dist_dir, :verbose => true)
FileUtils.cp_r("#{glare_core_repos_dir}/opengl/gl_data", dist_dir, :verbose => true)

# Copy library_licences.txt
FileUtils.cp_r("../library_licences.txt", dist_dir, :verbose => true)

# Copy LICENCE
FileUtils.cp_r("../LICENCE", dist_dir, :verbose => true)

# Copy about.txt
FileUtils.cp_r("../about.txt", dist_dir, :verbose => true)


# Make zip of directory
zip_path = "TerrainGen_v0.2.zip"

# Remove existing zip if present
FileUtils.rm(zip_path, :verbose=>true) if File.exist?(zip_path)

# -mx9: level 9 compression (max_
# -tzip : zip archive type
exec_command("sevenz a -mx9 -tzip \"#{zip_path}\" #{dist_dir}/*")
