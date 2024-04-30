
def filename_without_extension(name: str) -> str:
    return name.rsplit( ".", 1 )[0]