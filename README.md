# application-layer

To test and execute the code from this repository you have to locally clone this repository next to the interconnection-network repository and the execution-unit repository.
Afterwards you can set the cluster nodes in the get_network_information() method and start from main.rs inside this repository.

The "OS" part of this layer happens inside job.rs while main.rs can be used to create interesting use cases.
Right now we have not implemented any useful applications but there is a lot of potential to play around with :)
At this point the most interesting use case might be image editing (e.g. vignetting and filtering)
