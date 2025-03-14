using Flux, LinearAlgebra, Statistics, Distributions, Random, SpecialFunctions
using JLD2, Pkg, JSON3
using ProgressBars, ProgressMeter


function camera_distord(grid, ccde_level=9200, gain_factor=0, readNoiseSTD = 1.9, Nbits = 8)
    """
    Adds camera distortion to hologram.

    grid: Hologram input (NxN) 
    ccde_level: The average number of electrons for the CCD (Kodak KAI 29050 sensor has 2e4) (Default is 9200)
    gain_factor: Camera gain in dB (amplification) (Default is 0)
    readNoiseSTD: Kodak sensor has 12 (Default is 1.9)
    Nbits: Number of bits in hologram (Default is 8)
    # device: To process on CPU or GPU
    """
    ccde_level = ccde_level_with_camera_gain(ccde_level = ccde_level, gain_factor = gain_factor)
    # tmp_im = round.(grid .*ccde_level*ccd_gain(ccde_level = ccde_level, Nbits = Nbits))
    im = (grid.*ccde_level)
    # print(typeof(im))
    im = rand.(Poisson.(im)) # Mean values are im 
    # print(typeof(im))
    im = im .+ randn(size(im)) .* readNoiseSTD
    # print(typeof(im))
    im = im.*ccd_gain()
    # print(typeof(im))
    im = round.(im)
    # diff_im = (im .- tmp_im)./(2^(Nbits - 1))
    im = max.(zeros(size(im)), min.(im, (2^Nbits-1).*ones(size(im))))
    # noise_level = np.sqrt(sum(diff_im**2)) #Unused
    im #, noise_level
end

function w_camera_distord(grid, ccde_level=9200, gain_factor=0, readNoiseSTD = 1.9, Nbits = 8)
    """
    Adds camera distortion to weighted hologram.

    grid: Hologram input (NxN) 
    ccde_level: The average number of electrons for the CCD (Kodak KAI 29050 sensor has 2e4) (Default is 9200)
    gain_factor: Camera gain in dB (amplification) (Default is 0)
    readNoiseSTD: Kodak sensor has 12 (Default is 1.9)
    Nbits: Number of bits in hologram (Default is 8)
    # device: To process on CPU or GPU
     """
    im = grid
    im = im.*(2^(Nbits-1))
    im = max.(zeros(size(im)), min.(im, (2^Nbits-1).*ones(size(im))))
    im
end

function ccd_gain(;ccde_level = 9200, Nbits = 8)
    """
    Calculate the gain such that the hologram results in an image at the half gray level, or half saturation.
    ccde_level: The average number of electrons for the CCD (Kodak KAI 29050 sensor has 2e4) (Default is 9200)
    Nbits: Number of bits in hologram (Default is 8)
    """
    gain = (2^Nbits*0.5)/ccde_level
    gain 
end
function gkern(l=5, sig=1.)
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = range(-(l-1)/2, (l-1)/2, length=l)
    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = exp.(-.5 * ax.^2 ./ (sig.^2))
    kernel = gauss .* gauss' # dyadisches produkt
    kernel
end 

function ccde_level_with_camera_gain(;ccde_level = 9200, gain_factor = 0)
    """
    Gain factor is given in dB, so convert the ccd e- level to the desired value (to account for heavy amplification)
    ccde_level: The average number of electrons for the CCD (Kodak KAI 29050 sensor has 2e4) (Default is 9200)
    gain_factor: Camera gain in dB (amplification) (Default is 0)
    """
    ccde_level =  ccde_level/10^(gain_factor/10)
    ccde_level
end 


function make_holo(X,Y,Z,R, α; λ =355*10^-9, N_x = 512, N_y = 512, weighted = false)
    """
    Makes hologram for given set of coordinates, sizes and weights.
    X: List, x coordinate of particles
    Y: List, y coordinate of particles
    Z: List, z coordinate of particles
    R: List, radius of particles
    α: List, weight for each particle
    λ: scalar, wavelength of light (Default is 355nm)
    N_x: Field of view, x direction (in pixels) (Default is 512)
    N_y: Field of view, y direction (in pixels) (Default is 512)
    device: cpu or cuda processing (Default is cpu)
    """

    dx = 3e-6
    k = 2π / λ
    ε = 2.2204e-16  # This is the same value used in matlab epsilon
    grid = ones(Complex{Float64},N_y, N_x, 1, 1)
    N_y_half, N_x_half = N_y ÷ 2, N_x ÷ 2
    
    for (x_j,y_j,z_j,r_j,α_j) in zip(X,Y,Z,R,α)
        for y in -N_y_half:N_y_half-1
            y_dist_squared = (y*dx-y_j)^2
            for x in -N_x_half:N_x_half-1
                ρ =  √((x*dx-x_j)^2 + y_dist_squared)
                grid[y+N_y_half+1,x+N_x_half+1] -= α_j*r_j/(2.0im*(ρ+ε))*besselj(1,k*r_j*ρ/z_j)*exp(k*1.0im*(ρ^2/(2*z_j)))
            end
        end
    end
    
    if weighted == false
        return avgpool(camera_distord(abs.(grid).^2))[:,:]
    else 
        return avgpool(w_camera_distord(abs.(grid).^2))[:,:]
    end
end


function mask_maker(X,Y,Z,R;N_x=512, N_y=512)
    """
    creates the target maps for (x,y) as well as z and d (not used right now)
    """
    dx = 3e-6
    dz = 1e-3
    dr = 1e-6
    X = Int16.(round.(X./dx./ds_factor .+N_x ÷ 2 ))
    Y = Int16.(round.(Y./dx./ds_factor .+N_y ÷ 2 )) 
  
    Z = (Z./dz)
    R = (R./dr)
    gk = gkern(3,1)

    
    mask = zeros(Float32,3,N_y+4,N_x+4) # Making the mask bigger here to take care of the particle at near (1pixel from edge) boundary
    
    for (x,y,z,r) in zip(X,Y,Z,R)
        mask[1, y+2:y+4, x+2:x+4] .= gk  # center of particle is (1+2+x,1+2+y) (+1 because x is zero indexed, julia arrays 1 indexed.+2 because mask was created with margin)
        mask[2, y+2:y+4, x+2:x+4] .= z
        mask[3, y+2:y+4, x+2:x+4] .= r
    end
    mask[:,3:386,3:386]
end 

function diffraction(X, Y, Z, R, α; λ = 355*1e-9, N_x = 512, N_y = 512)
    hologram = make_holo(X, Y, Z, R, α, λ = λ, N_x = N_x, N_y = N_y, weighted = false) #Always making holograms at original resolve and then ds. 
    mask = (X./dx, Y./dx, Z./dz, R./dr)
    weighted_hologram = make_holo(X, Y, Z, [50.0 * dr for _ in 1:N_particles], α, λ = λ/8, N_x = N_x, N_y = N_y, weighted = true)
    hologram, mask, weighted_hologram 
end 

seed = 42
dx = 3e-6
dz = 1e-3
dr = 1e-6
λ = 355e-9
# λ = 532e-9
N_x = 2048
N_y = 2048
ds_factor = 2
zmax = 200
zmin = 5
rmax = 50
rmin = 3
N_particles = 32*16
N = 40|> Int # Number of holograms 

save_base_dir = "/project.lmp/cloudkite-proc/2048x2048_data/"
holograms_file_name = "holograms_synthetic_17k_32x9Particles_set6_uniform_halfres"
masks_file_name = "masks_synthetic_17k_32x9Particles_set6_uniform_halfres"
weighted_holograms_file_name = "weighted_holograms_synthetic_17k_32x9Particles_set6_uniform_halfres"
format = ".jld2"


# Storage for holograms, masks, and coordinates
print("Let's make Holograms!")
avgpool = MeanPool((ds_factor,ds_factor))

holograms = []
masks = []
weighted_holograms = []


function genHolo()
    X = [(rand() * (N_x//1 - 2) - (N_x//2 - 2)) * dx for _ in 1:N_particles]
    Y = [(rand() * (N_y//1 - 2) - (N_y//2 - 2)) * dx for _ in 1:N_particles]
    z_arg = [rand() * (zmax-zmin) + zmin for _ in 1:N_particles]
    r_arg = [rand() * (rmax-rmin) + rmin for _ in 1:N_particles]
    # r_arg = [min(-log2(rand()) * rmin * 2 + rmin, rmax) for _ in 1:N_particles]
    Z = [dz * z for z in z_arg]
    R = [dr * r for r in r_arg]
    α = [1.0 for _ in 1:N_particles]
    data = diffraction(X, Y, Z, R, α, λ = λ, N_x = N_x, N_y = N_y)
    return data
end

function execute_holo_creation()
    # Initialize thread_data with tuples of empty arrays
    numThreads = Threads.nthreads()
    println(numThreads)
    thread_data = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}, Vector{Matrix{Float64}}}}(undef, numThreads)
    Threads.@threads for i in 1:numThreads
        tId = Threads.threadid()
        thread_data[tId] = (Vector{Matrix{Float64}}(), Vector{Array{Float32, 3}}(), Vector{Matrix{Float64}}())
    end
    
    @time Threads.@threads :static for t in ProgressBar(1:N)
        tId = Threads.threadid()
        data = genHolo()  # Assuming you have a function genHolo() to generate data
        push!(thread_data[tId][1], data[1])  # Push data into the first array
        push!(thread_data[tId][2], data[2])  # Push data into the second array
        push!(thread_data[tId][3], data[3])  # Push data into the third array
    end

    return thread_data
end


println("Let's Begin...")

@time thread_data = execute_holo_creation()

holograms = []
masks = []
weighted_holograms = []
for tdata in tqdm(thread_data)
    for holo in tdata[1]
        push!(holograms, holo)
    end
    for mask in tdata[2]
        push!(masks, mask)
    end
    for wholo in tdata[3]
        push!(weighted_holograms, wholo)
    end
end

println(typeof(holograms), typeof(masks), size(weighted_holograms))

ds_size = convert(Int, N_x/ds_factor)
holos = zeros(Float16,N, ds_size, ds_size)
maps = []
wholos = zeros(Float16, N, ds_size, ds_size)

for t in tqdm(1:size(holos)[1])
    holos[t, :, :] = holograms[t]
    push!(maps, collect(masks[t]))
    wholos[t, :, :] = weighted_holograms[t]
end


holograms = convert(Array{Float16}, holos)
masks = maps
weighted_holograms = convert(Array{Float16}, wholos)
println(size(holograms))
println(typeof(holograms))
println(Base.summarysize(holograms)/1024/1024/1024)
println(size(masks))
println(typeof(masks))
println(Base.summarysize(masks)/1024/1024/1024)
println(size(weighted_holograms))
println(typeof(weighted_holograms))
println(Base.summarysize(weighted_holograms)/1024/1024/1024)



save_object(save_base_dir*holograms_file_name*format, holograms)
JSON3.write(save_base_dir*masks_file_name*".json", masks)
save_object(save_base_dir*weighted_holograms_file_name*format, weighted_holograms)


