import launch

if not launch.is_installed("tensorflow"):
    launch.run_pip("install tensorflow", "requirements for wd14-tagger")
