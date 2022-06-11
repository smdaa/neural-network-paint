using Flux
using Images
using VideoIO
using FileIO
using Printf

function build_model(n, activation_fn)
    return Chain(Dense(3, n, activation_fn, bias=true, init=Flux.truncated_normal), 
        
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        
        #Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        #Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        #Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        #Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal), 
        
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal),
        Dense(n, n, activation_fn, bias=false, init=Flux.truncated_normal), 
        
        Dense(n, 3, sigmoid, bias=false, init=Flux.truncated_normal),
    )
end


function generate_image(model, size_h, size_w, zoom)
    image = zeros(3, size_h, size_w)

    Threads.@threads for i in 1:size_h
        for j in 1:size_w
            color = model([(i / size_h - 0.5) * zoom, (j / size_w - 0.5) * zoom, sqrt((i / size_h - 0.5)^2 + (j / size_w - 0.5)^2) * zoom])
            image[:, i, j] = color
        end
    end
    return image
end

function generate_zooming_images(model, size_h, size_w, num_images)
    zoom = 1
    for i in 1:num_images

        @time image = generate_image(model, size_h, size_w, zoom)
        name = @sprintf("images/%05d.png", i)
        save(name, colorview(RGB, image))

        zoom = zoom * 0.95

    end

    imgnames = filter(x -> occursin(".png", x), readdir("images/"))
    intstrings = map(x -> split(x, ".")[1], imgnames)
    p = sortperm(parse.(Int, intstrings))
    
    imgstack = []
    for imgname in imgnames[p]
        push!(imgstack, load("images/" * imgname))
    end
    
    encoder_options = (crf=23, preset="medium")
    VideoIO.save("video.mp4", imgstack, framerate=24, encoder_options=encoder_options)
end

model = build_model(25, tanh)
size_h = 2000
size_w = 2000
num_images = 128

print(Threads.nthreads())

generate_zooming_images(model, size_h, size_w, num_images)
