import os
import time
import shutil


def atomic_copy(src_path, cache_path, max_age_in_sec=60, num_trials=10):
    cache_path = os.path.expandvars(cache_path) # expand $SLURM_JOB_ID or $USER
    if os.path.normpath(src_path) == os.path.normpath(cache_path): 
        return cache_path # nothing to do
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    try:
        # atomic way of checking if file exist
        fdest = os.open(cache_path,  os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        is_master = True

        with os.fdopen(fdest, 'wb') as dst_file:
            for _ in range(num_trials):
                try:
                    if callable( src_path ):
                        # fake source file, it doesn't actually exist 
                        src_path( dst_file ) # create file directly
                    else:
                        with open(src_path, 'rb') as src_file:
                            # print('copying',src_path,'-->',cache_path)
                            shutil.copyfileobj(src_file, dst_file)
                        break
                except IOError:
                    time.sleep(1)
            else:
                raise IOError(f"Tried {num_trials} times to copy {src_file} --> {dst_file} but couldn't")

            # to signal that copy is finished, set the file read-only
            os.chmod(fdest, 0o444)

    except FileExistsError:
        # file exist already, waiting for the copy to finish in another process
        is_master = False

        while ((fs:=os.stat(cache_path)).st_mode & 0o777) != 0o444:
            time.sleep(10e-6) # wait for 10 micro-seconds

            if max_age_in_sec > 0 and (time.time() - fs.st_mtime) > max_age_in_sec:
                # target file is not modified since long, 
                # so most likely nobody's taking care of the copy
                is_master = atomic_copy(src_path, cache_path+'.tmp')

                # only one process will do that
                if is_master:
                    # replace the file
                    shutil.copyfile(cache_path+'.tmp', cache_path) 
                    # erase the temp file
                    os.remove(cache_path+'.tmp')
                    # signal the end
                    os.chmod(cache_path, 0o444) 

    return is_master
