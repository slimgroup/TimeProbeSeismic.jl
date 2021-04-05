if VERSION<v"1.6"
    Pkg.rm("JUDI")
    Pkg.add(name="JUDI", version="2.2.0", rev="pyload")
    Pkg.resolve()
    Pkg.instantiate()
end
