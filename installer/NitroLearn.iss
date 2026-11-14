; NitroLearn's setup wizard, NitroLearn-Setup.exe. The release workflow (.github/workflows/release.yml, job
; "setup") builds it on Windows with Inno Setup (free for any use; https://jrsoftware.org/isinfo.php):
;     python tools\setup_wizard.py
;     ISCC /DAppVersion=<the version in Cargo.toml> installer\NitroLearn.iss
; It carries NitroLearn's own files (installer\build\app) and uv. Everything else downloads while it
; installs, done by tools\install_steps.py through install.ps1, exactly as "Install NitroLearn.cmd" does for
; the zip: this script only asks the questions and shows the progress.

#ifndef AppVersion
  #error Give the version: ISCC /DAppVersion=<the version in Cargo.toml> NitroLearn.iss
#endif
; The optional parts, as tools\install_steps.py names them (PARTS): the Components page's names.
#define Parts "gpu,rlbot,start_menu,desktop,rust,dev,recordings"

[Setup]
AppId={{8C1E5A42-3B7D-4F2A-9E61-6D0B7C2F4A19}
AppName=NitroLearn
AppVersion={#AppVersion}
AppVerName=NitroLearn {#AppVersion}
AppPublisher=Raze
AppPublisherURL=https://nitrolearn.net
AppSupportURL=https://discord.gg/bYP3kYtryg
AppUpdatesURL=https://github.com/AverageCoderx/NitroLearn/releases
; One folder per user, no administrator rights; nothing outside it but the chosen shortcuts.
DefaultDirName={localappdata}\Programs\NitroLearn
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=auto
; The folder is checked by the script (NextButtonClick): one question at most, and none for a folder an
; uninstall left with the user's bots in it.
DirExistsWarning=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
WizardImageFile=wizard-100.png,wizard-125.png,wizard-150.png,wizard-200.png,wizard-250.png
WizardSmallImageFile=wizard-small-100.png,wizard-small-125.png,wizard-small-150.png,wizard-small-200.png,wizard-small-250.png
WizardImageStretch=no
WizardImageBackColor=$131414
SetupIconFile=..\python\nitro\brand\icon.ico
LicenseFile=build\license.rtf
OutputDir=build\output
OutputBaseFilename=NitroLearn-Setup
Compression=lzma2/max
SolidCompression=yes
UninstallDisplayName=NitroLearn
UninstallDisplayIcon={app}\python\nitro\brand\icon.ico
; The uninstall entry in Windows' Apps list (one per-user key under HKCU, removed by uninstalling). Set it to
; "no" for an install that writes nothing to the registry: uninstall is then unins000.exe in the folder.
CreateUninstallRegKey=yes
; With PrivilegesRequired=lowest Inno Setup passes its RedirectionGuard mitigation on to the programs it runs
; (uv, cargo, the RLBot installer); a per-user install gains nothing from it, so it is off.
RedirectionGuard=no
; Unsigned for now: Windows SmartScreen shows "More info > Run anyway". With a code-signing certificate,
; name a sign tool when compiling (ISCC "/Snitrosign=signtool sign /fd sha256 /tr <timestamp url> /td sha256 $f")
; and uncomment these two lines:
;SignTool=nitrosign
;SignedUninstaller=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer: train your own Rocket League bots, watch them play in 3D, and play them in RLBot.%n%nEverything goes into one folder that is yours. Your other programs and any Python you already have are not touched.%n%nSetup downloads about 3.5 GB, mostly PyTorch for your graphics card, and needs about 10 GB of free disk space with the recommended parts (about 2 GB without the graphics-card PyTorch). It takes about 20 minutes on a normal connection.
SelectDirLabel3=Setup will install [name] into the following folder. Your bots and settings are kept inside it too.
DiskSpaceGBLabel=NitroLearn needs about 10 GB of free disk space with the recommended parts (about 2 GB without the graphics-card PyTorch). A drive with lots of space (like D:) is a good choice if C: is nearly full.
DiskSpaceMBLabel=NitroLearn needs about 10 GB of free disk space with the recommended parts (about 2 GB without the graphics-card PyTorch). A drive with lots of space (like D:) is a good choice if C: is nearly full.
; Asked before the question about keeping the bots (the uninstall code), so it promises no full removal.
ConfirmUninstall=Uninstall %1?%n%nNext, you choose whether to keep your bots and settings. RLBot is a separate program and is not touched.
; Kept bots are named in their own message after this one (usDone), only when they were kept.
UninstalledAll=%1 is uninstalled.
SelectComponentsDesc=Which parts of NitroLearn should be installed?
FinishedLabelNoIcons=NitroLearn is installed. Open it from the Start menu any time.%n%nThe first time it opens, a short welcome walks you through making your first bot.%n%nTo uninstall, use Windows' Settings > Apps. It asks whether to keep your bots first.
FinishedLabel=NitroLearn is installed. Open it from the Start menu any time.%n%nThe first time it opens, a short welcome walks you through making your first bot.%n%nTo uninstall, use Windows' Settings > Apps. It asks whether to keep your bots first.

[Types]
Name: "recommended"; Description: "Recommended installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

; The installer's menu (tools\install_steps.py), with the same parts and defaults. Sizes are on disk in the
; folder, the downloads kept in .cache included: "main" holds everything always installed, with PyTorch for
; the processor (about 2 GB); "gpu" what PyTorch for the graphics card adds (about 9 GB in place of 1.5), so
; the recommended parts come to about 9.6 GB. RLBot installs itself outside the folder.
[Components]
Name: "main"; Description: "NitroLearn, its own Python and the arena files"; Types: recommended custom; Flags: fixed; ExtraDiskSpaceRequired: 2147483648
Name: "gpu"; Description: "PyTorch for the graphics card (off: the slower processor version)"; Types: recommended; ExtraDiskSpaceRequired: 8053063680
Name: "rlbot"; Description: "RLBot v5, to play your bots in Rocket League (its own installer opens)"; Types: recommended
Name: "start_menu"; Description: "Start menu shortcut"; Types: recommended
Name: "desktop"; Description: "Desktop shortcut"
Name: "rust"; Description: "Tools to build your own Rust rewards"; ExtraDiskSpaceRequired: 1610612736
Name: "dev"; Description: "Developer tools (tests, rustfmt, clippy)"; ExtraDiskSpaceRequired: 1717986918
Name: "recordings"; Description: "RocketSim's game recordings now (else downloaded the first time a feature needs them)"; ExtraDiskSpaceRequired: 124780544

; The note an uninstall leaves beside the bots it kept goes once NitroLearn is in the folder again.
[InstallDelete]
Type: files; Name: "{app}\Kept by NitroLearn.txt"

[Files]
Source: "build\app\*"; DestDir: "{app}"; Components: main; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "build\uv\*"; DestDir: "{app}\.tools\uv"; Components: main; Flags: ignoreversion

[Run]
; Windowless, as the Start menu shortcut starts it (tools\app_shortcuts.ps1).
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m nitro.dashboard.winapp open"; WorkingDir: "{app}"; Description: "Launch NitroLearn"; Flags: postinstall nowait skipifsilent; Check: InstallSucceeded

[UninstallRun]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\app_shortcuts.ps1"" -Remove"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveShortcuts"

; What the install made in the folder, and NitroLearn's own folders (an update can add files to them). The
; user's own folders (UserFolders below) go only when the uninstaller is told not to keep them.
[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\.tools"
Type: filesandordirs; Name: "{app}\.cache"
Type: filesandordirs; Name: "{app}\.tmp"
Type: filesandordirs; Name: "{app}\collision_meshes"
Type: filesandordirs; Name: "{app}\rocketsim_recordings"
Type: filesandordirs; Name: "{app}\target"
Type: filesandordirs; Name: "{app}\build"
Type: filesandordirs; Name: "{app}\release-files"
Type: filesandordirs; Name: "{app}\.github"
Type: filesandordirs; Name: "{app}\LICENSES"
Type: filesandordirs; Name: "{app}\Shortcuts"
Type: filesandordirs; Name: "{app}\benchmarks"
Type: filesandordirs; Name: "{app}\configs"
Type: filesandordirs; Name: "{app}\crates"
Type: filesandordirs; Name: "{app}\docs"
Type: filesandordirs; Name: "{app}\installer"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\tests"
Type: filesandordirs; Name: "{app}\tools"
Type: files; Name: "{app}\*"

[Code]
const
  UserFolders = 'runs,my-settings,packs,replays,bots,ab,wandb,saves';
  { Written by an uninstall that keeps the user's folders, beside them. }
  KeptNote = 'Kept by NitroLearn.txt';
  DirectoryAttribute = $10;
  { RLBot v5's programs and their folder under %LOCALAPPDATA% or its Programs folder, as
    python\nitro\rlbot_find.py finds them: only RLBot v5 counts, an older RLBot does not. }
  RLBotPrograms = 'launcher.exe,rlbotgui.exe,RLBotServer.exe';
  RLBotFolder = 'RLBot5\bin\';
  { The Components page's lines, in [Components] order. }
  GpuItem = 1;
  RLBotItem = 2;
  MarkPending = #$00B7;
  MarkRunning = #$2026;
  MarkDone = #$2713;
  MarkFailed = #$2717;

var
  StepKeys, StepTexts, StepStates: TArrayOfString;
  Checklist: TNewMemo;
  LogFile, StopText, DoctorText, NoteText: String;
  Succeeded, CardShown, RLBotInstalled, NoCard, NoDriver, KeepUserFiles: Boolean;
  TypesChangedBefore: TNotifyEvent;

{ ---------- the install folder ---------- }

function IsUserFolder(const Name: String): Boolean;
begin
  Result := Pos(',' + Lowercase(Name) + ',', ',' + UserFolders + ',') > 0;
end;

{ An empty folder, or one an uninstall left: only the user's own folders and the note about them. }
function HoldsOnlyKeptFiles(const Dir: String): Boolean;
var
  Found: TFindRec;
begin
  Result := True;
  if FindFirst(AddBackslash(Dir) + '*', Found) then
  try
    repeat
      if (Found.Name <> '.') and (Found.Name <> '..') then
      begin
        if (Found.Attributes and DirectoryAttribute) <> 0 then
          Result := IsUserFolder(Found.Name)
        else
          Result := CompareText(Found.Name, KeptNote) = 0;
      end;
    until (not Result) or (not FindNext(Found));
  finally
    FindClose(Found);
  end;
end;

{ Uninstalling removes the folder's contents, so it must be NitroLearn's own: new, empty, NitroLearn, or what
  an uninstall kept of it. }
function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Dir := WizardDirValue;
    if DirExists(Dir) and not HoldsOnlyKeptFiles(Dir) and not FileExists(AddBackslash(Dir) + 'Install NitroLearn.cmd') then
    begin
      MsgBox('This folder already holds other files. Choose a new or empty folder for NitroLearn: uninstalling NitroLearn later removes its folder.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

{ ---------- a NitroLearn that is running ---------- }

{ The programs running from Dir (NitroLearn's Python, the dashboard, a training run in the background, an
  exported bot), as a list of names; '' when none. Setup's own uninstaller is left out. }
function RunningFrom(const Dir: String): String;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  Prefix, Path, Name: String;
begin
  Result := '';
  Prefix := Lowercase(AddBackslash(Dir));
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT Name, ExecutablePath FROM Win32_Process');
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      if not VarIsNull(Item.ExecutablePath) and not VarIsNull(Item.Name) then
      begin
        Path := Item.ExecutablePath;  { a Variant: stored in a String before any String function }
        Path := Lowercase(Path);
        Name := Item.Name;
        if (Copy(Path, 1, Length(Prefix)) = Prefix) and (Pos('unins', Lowercase(Name)) <> 1) and (Pos(', ' + Lowercase(Name) + ',', ', ' + Lowercase(Result) + ',') = 0) then
        begin
          if Result <> '' then
            Result := Result + ', ';
          Result := Result + Name;
        end;
      end;
    end;
  except
    Result := '';  { Windows could not say: nothing is blocked }
  end;
end;

{ Asks to stop NitroLearn while any of its programs run: True once none does, False when the user cancels. }
function NitroLearnStopped(const Dir, Doing: String): Boolean;
var
  Running: String;
begin
  Result := True;
  Running := RunningFrom(Dir);
  while Result and (Running <> '') do
  begin
    Result := SuppressibleMsgBox('NitroLearn is still running (' + Running + '), so ' + Doing + ' would fail half way.' + #13#10#13#10 + 'Close the dashboard, and stop any bot that is training: training goes on in the background after the window closes, so open the dashboard and press Stop and save on each training bot (or restart the PC).' + #13#10#13#10 + 'Then press Retry.', mbError, MB_RETRYCANCEL, IDCANCEL) = IDRETRY;
    if Result then
      Running := RunningFrom(Dir);
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if DirExists(WizardDirValue) and not NitroLearnStopped(WizardDirValue, 'installing') then
    Result := 'NitroLearn is still running. Stop it, then run NitroLearn-Setup again.';
end;

function InitializeUninstall: Boolean;
begin
  Result := NitroLearnStopped(ExpandConstant('{app}'), 'uninstalling');
end;

{ ---------- the Components page: the graphics card ---------- }

function NvidiaCard: String;
var
  Output: TExecOutput;
  Code, I: Integer;
  Places: TArrayOfString;
begin
  Result := '';
  SetArrayLength(Places, 2);
  Places[0] := ExpandConstant('{sys}\nvidia-smi.exe');
  Places[1] := ExpandConstant('{commonpf64}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');
  for I := 0 to GetArrayLength(Places) - 1 do
    if (Result = '') and FileExists(Places[I]) then
    try
      if ExecAndCaptureOutput(Places[I], '--query-gpu=name --format=csv,noheader', '', SW_HIDE, ewWaitUntilTerminated, Code, Output) then
        if (Code = 0) and (GetArrayLength(Output.StdOut) > 0) then
          Result := Trim(Output.StdOut[0]);
    except
      Result := '';
    end;
end;

{ An NVIDIA card Windows lists (also one running Windows' basic driver) when nvidia-smi gave no answer: its
  driver is missing or too old. VEN_10DE is NVIDIA's id in a device id. }
function NvidiaAdapterFound: Boolean;
var
  Output: TExecOutput;
  Code, I: Integer;
begin
  Result := False;
  try
    if ExecAndCaptureOutput(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), '-NoProfile -NonInteractive -Command "Get-CimInstance Win32_VideoController | ForEach-Object { [string]$_.Name + [string]$_.PNPDeviceID }"', '', SW_HIDE, ewWaitUntilTerminated, Code, Output) then
      if Code = 0 then
        for I := 0 to GetArrayLength(Output.StdOut) - 1 do
          if (Pos('VEN_10DE', Uppercase(Output.StdOut[I])) > 0) or (Pos('NVIDIA', Uppercase(Output.StdOut[I])) > 0) then
            Result := True;
  except
    Result := False;
  end;
end;

function RLBotV5Found: Boolean;
var
  Bases: TArrayOfString;
  Rest, Name: String;
  Comma, I: Integer;
begin
  Result := False;
  SetArrayLength(Bases, 2);
  Bases[0] := ExpandConstant('{localappdata}');
  Bases[1] := ExpandConstant('{localappdata}\Programs');
  Rest := RLBotPrograms + ',';
  while (Rest <> '') and not Result do
  begin
    Comma := Pos(',', Rest);
    Name := Copy(Rest, 1, Comma - 1);
    Rest := Copy(Rest, Comma + 1, Length(Rest));
    for I := 0 to GetArrayLength(Bases) - 1 do
      if FileExists(AddBackslash(Bases[I]) + RLBotFolder + Name) then
        Result := True;
  end;
end;

{ A part this PC settles by itself: unticked and greyed, its line saying why. }
procedure SettleItem(Index: Integer; const Caption: String);
begin
  WizardForm.ComponentsList.ItemCaption[Index] := Caption;
  WizardForm.ComponentsList.Checked[Index] := False;
  WizardForm.ComponentsList.ItemEnabled[Index] := False;
end;

procedure SettleFoundParts;
begin
  if NoCard and NoDriver then
    SettleItem(GpuItem, 'PyTorch for the graphics card: this PC has an NVIDIA card, but its driver is missing or does not answer, so the processor version is installed. Install the driver from nvidia.com/drivers, restart, then run this setup again')
  else if NoCard then
    SettleItem(GpuItem, 'PyTorch for the graphics card: no NVIDIA card was found, so the processor version is installed');
  if RLBotInstalled then
    SettleItem(RLBotItem, 'RLBot v5: already installed on this PC, NitroLearn finds it by itself');
end;

procedure ShowCard;
var
  Card: String;
begin
  CardShown := True;
  if WizardForm.ComponentsList.Items.Count <= RLBotItem then
    Exit;
  Card := NvidiaCard;
  NoCard := Card = '';
  NoDriver := NoCard and NvidiaAdapterFound;
  if not NoCard then
    WizardForm.ComponentsList.ItemCaption[GpuItem] := 'PyTorch for the graphics card, ' + Card + ' (off: the slower processor version)';
  RLBotInstalled := RLBotV5Found;
  SettleFoundParts;
end;

{ Choosing a type ticks its parts again: the ones this PC settled stay as they are. }
procedure TypesChanged(Sender: TObject);
begin
  TypesChangedBefore(Sender);
  if CardShown then
    SettleFoundParts;
end;

{ ---------- the install: tools\install_steps.py through install.ps1, with a checklist ---------- }

function SelectedParts: String;
var
  Rest, Name: String;
  Comma: Integer;
begin
  Result := '';
  Rest := '{#Parts},';
  while Rest <> '' do
  begin
    Comma := Pos(',', Rest);
    Name := Copy(Rest, 1, Comma - 1);
    Rest := Copy(Rest, Comma + 1, Length(Rest));
    if WizardIsComponentSelected(Name) then
    begin
      if Result <> '' then
        Result := Result + ',';
      Result := Result + Name;
    end;
  end;
  if Result = '' then
    Result := 'none';
end;

procedure AddStep(const Key, Text: String);
var
  N: Integer;
begin
  N := GetArrayLength(StepKeys);
  SetArrayLength(StepKeys, N + 1);
  SetArrayLength(StepTexts, N + 1);
  SetArrayLength(StepStates, N + 1);
  StepKeys[N] := Key;
  StepTexts[N] := Text;
  StepStates[N] := '';
end;

{ The parts in the order install_steps.py does them; its "@@nitro <key> <state>" lines tick them. }
procedure ListSteps;
begin
  AddStep('pc', 'Checked this PC');
  AddStep('python', 'Python 3.11 in NitroLearn''s folder (nothing else on the PC changed)');
  if WizardIsComponentSelected('gpu') then
    AddStep('torch', 'PyTorch for the graphics card')
  else
    AddStep('torch', 'PyTorch for the processor');
  AddStep('core', 'NitroLearn''s core and programs');
  if WizardIsComponentSelected('rust') or WizardIsComponentSelected('dev') then
    AddStep('rust', 'Rust in NitroLearn''s folder');
  AddStep('meshes', 'The arena files (from RLGym, hash checked)');
  if WizardIsComponentSelected('recordings') then
    AddStep('recordings', 'RocketSim''s game recordings');
  AddStep('check', 'Final check of everything');
  if WizardIsComponentSelected('desktop') then
    AddStep('shortcuts', 'Start menu and desktop shortcuts')
  else if WizardIsComponentSelected('start_menu') then
    AddStep('shortcuts', 'Start menu shortcut');
  if WizardIsComponentSelected('rlbot') then
    AddStep('rlbot', 'RLBot v5 (its own installer opens)');
end;

procedure ShowSteps;
var
  I, Done: Integer;
  Shown, Mark: String;
begin
  Shown := 'This takes about 20 minutes on a normal connection, longer on a slow one. It cannot be' + #13#10 + 'cancelled part way: let it finish (what was downloaded is kept if it stops).' + #13#10#13#10;
  Done := 0;
  for I := 0 to GetArrayLength(StepKeys) - 1 do
  begin
    Mark := MarkPending;
    if StepStates[I] = 'start' then
      Mark := MarkRunning
    else if StepStates[I] = 'done' then
    begin
      Mark := MarkDone;
      Done := Done + 1;
    end
    else if StepStates[I] = 'fail' then
    begin
      Mark := MarkFailed;
      Done := Done + 1;
    end;
    Shown := Shown + Mark + ' ' + StepTexts[I] + #13#10;
  end;
  Checklist.Text := Shown;
  WizardForm.ProgressGauge.Position := Done;
end;

procedure SetStep(const Line: String);
var
  Space, I: Integer;
  Key, State: String;
begin
  Space := Pos(' ', Line);
  if Space = 0 then
    Exit;
  Key := Copy(Line, 1, Space - 1);
  State := Trim(Copy(Line, Space + 1, Length(Line)));
  for I := 0 to GetArrayLength(StepKeys) - 1 do
    if StepKeys[I] = Key then
      StepStates[I] := State;
  ShowSteps;
end;

procedure InstallOutput(const S: String; const Error, FirstLine: Boolean);
var
  Line: String;
begin
  Line := Trim(S);
  SaveStringToFile(LogFile, S + #13#10, True);
  if Error then
    Exit;
  if Copy(Line, 1, 8) = '@@nitro ' then
  begin
    SetStep(Copy(Line, 9, Length(Line)));
    Exit;
  end;
  if Copy(Line, 1, 4) = '==> ' then
    WizardForm.StatusLabel.Caption := Copy(Line, 5, Length(Line)) + '...'
  else if Line <> '' then
    WizardForm.FilenameLabel.Caption := Line;
  if Copy(Line, 1, 15) = 'SETUP STOPPED: ' then
    StopText := Copy(Line, 16, Length(Line));
  if Copy(Line, 1, 12) = 'SETUP NOTE: ' then
    NoteText := NoteText + Copy(Line, 13, Length(Line)) + #13#10#13#10;
  if Copy(Line, 1, 28) = 'NitroLearn is installed, but' then
    DoctorText := Line;
end;

{ The release's files to install from: beside the setup (the release's files downloaded together and
  unzipped, NitroLearn-Setup.exe among them), else in a release-files folder beside it; '' for none. A
  folder that ends in a backslash (a drive) gets a dot, so the backslash cannot escape the closing quote. }
function ReleaseFilesFolder: String;
var
  Src: String;
begin
  Result := '';
  Src := ExpandConstant('{src}');
  if FileExists(AddBackslash(Src) + 'SHA256SUMS.txt') then
    Result := Src
  else if FileExists(AddBackslash(Src) + 'release-files\SHA256SUMS.txt') then
    Result := AddBackslash(Src) + 'release-files';
  if (Result <> '') and (Copy(Result, Length(Result), 1) = '\') then
    Result := Result + '.';
end;

{ How to open NitroLearn's folder: Windows hides AppData in its file windows, so the folder's address is
  given to paste into Run. }
function OpenFolderHint: String;
begin
  Result := 'To open the folder, press the Windows key and R together, paste this and press Enter:' + #13#10 + ExpandConstant('{app}');
end;

{ A step's state in the checklist: '', start, done or fail. }
function StepState(const Key: String): String;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to GetArrayLength(StepKeys) - 1 do
    if StepKeys[I] = Key then
      Result := StepStates[I];
end;

procedure RunInstallSteps;
var
  Params, Folder: String;
  Code, I: Integer;
begin
  LogFile := ExpandConstant('{app}\install-log.txt');
  DeleteFile(LogFile);
  ListSteps;
  WizardForm.ProgressGauge.Min := 0;
  WizardForm.ProgressGauge.Max := GetArrayLength(StepKeys);
  Checklist.Visible := True;
  ShowSteps;
  WizardForm.CancelButton.Enabled := False;
  WizardForm.StatusLabel.Caption := 'Downloading and setting up NitroLearn''s parts...';
  WizardForm.FilenameLabel.Caption := '';
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '" -Parts "' + SelectedParts + '" -NoLaunch -Progress';
  { The release's files beside the setup: installed from them before anything is downloaded (offline, or a
    trial build). }
  Folder := ReleaseFilesFolder;
  if Folder <> '' then
    Params := Params + ' -Bundles "' + Folder + '"';
  Code := -1;
  try
    if not ExecAndLogOutput(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, Code, @InstallOutput) then
      StopText := 'Windows PowerShell could not be started (' + SysErrorMessage(Code) + ').';
  except
    StopText := GetExceptionMessage;
  end;
  Succeeded := Code = 0;
  if Succeeded then
  begin
    for I := 0 to GetArrayLength(StepKeys) - 1 do
      if StepStates[I] = '' then
        StepStates[I] := 'done';  { nothing to do: it was there already }
    ShowSteps;
  end
  else
  begin
    for I := 0 to GetArrayLength(StepKeys) - 1 do
      if StepStates[I] = 'start' then
        StepStates[I] := 'fail';  { the step it stopped in }
    ShowSteps;
  end;
  { What the install could not do as chosen (PyTorch for the processor in place of the graphics card's,
    the driver to update): said once it is done. }
  if (Succeeded or (DoctorText <> '')) and (NoteText <> '') then
    SuppressibleMsgBox('NitroLearn is installed. Please read this:' + #13#10#13#10 + NoteText + 'The whole log is install-log.txt in NitroLearn''s folder. ' + OpenFolderHint, mbInformation, MB_OK, IDOK);
  if not Succeeded then
  begin
    if DoctorText <> '' then
      SuppressibleMsgBox('NitroLearn is installed, but its final check found something to fix.' + #13#10#13#10 + 'Open NitroLearn (Finish opens it) and press "Check this PC" at the top: it says what to fix.' + #13#10#13#10 + 'The whole log is install-log.txt in NitroLearn''s folder. ' + OpenFolderHint, mbInformation, MB_OK, IDOK)
    else
    begin
      if StopText = '' then
        StopText := 'Setup did not finish.';
      SuppressibleMsgBox('Setup stopped: ' + StopText + #13#10#13#10 + 'What was downloaded is kept: fix this, then run NitroLearn-Setup again.' + #13#10#13#10 + 'The whole log is install-log.txt in NitroLearn''s folder. ' + OpenFolderHint, mbError, MB_OK, IDOK);
    end;
  end;
  WizardForm.CancelButton.Enabled := True;
end;

{ NitroLearn opens after a good install, or one whose final check only found something to fix. }
function InstallSucceeded: Boolean;
begin
  Result := Succeeded or (DoctorText <> '');
end;

procedure InitializeWizard;
begin
  Checklist := TNewMemo.Create(WizardForm);
  Checklist.Parent := WizardForm.InstallingPage;
  Checklist.Left := WizardForm.ProgressGauge.Left;
  Checklist.Width := WizardForm.ProgressGauge.Width;
  Checklist.Top := WizardForm.ProgressGauge.Top + WizardForm.ProgressGauge.Height + ScaleY(12);
  Checklist.Height := WizardForm.InstallingPage.ClientHeight - Checklist.Top - ScaleY(4);
  Checklist.ReadOnly := True;
  Checklist.ScrollBars := ssVertical;
  Checklist.Font.Name := 'Consolas';
  Checklist.Visible := False;
  TypesChangedBefore := WizardForm.TypesCombo.OnChange;
  WizardForm.TypesCombo.OnChange := @TypesChanged;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectComponents) and not CardShown then
    ShowCard;
  { A stopped install says so; with no Start menu shortcut (not chosen, or it could not be made), the
    folder's own one and how to get there. }
  if (CurPageID = wpFinished) and not InstallSucceeded then
    WizardForm.FinishedLabel.Caption := 'Setup stopped before NitroLearn was fully installed (the message before this page said why). What was downloaded is kept: fix that, then run NitroLearn-Setup again and choose the same folder.'
  else if (CurPageID = wpFinished) and (StepState('shortcuts') <> 'done') then
    WizardForm.FinishedLabel.Caption := 'NitroLearn is installed. Open it with "Open the dashboard" in its Shortcuts folder. ' + OpenFolderHint + #13#10#13#10 + 'The first time it opens, a short welcome walks you through making your first bot.' + #13#10#13#10 + 'To uninstall, use Windows'' Settings > Apps. It asks whether to keep your bots first.';
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    RunInstallSteps;
end;

{ ---------- uninstall: the user's own folders stay unless they say otherwise ---------- }

{ The note left beside the folders an uninstall kept, so the folder says what it holds. }
procedure WriteKeptNote(const App: String);
var
  Rest, Name, Kept: String;
  Comma: Integer;
begin
  Kept := '';
  Rest := UserFolders + ',';
  while Rest <> '' do
  begin
    Comma := Pos(',', Rest);
    Name := Copy(Rest, 1, Comma - 1);
    Rest := Copy(Rest, Comma + 1, Length(Rest));
    if DirExists(AddBackslash(App) + Name) then
    begin
      if Kept <> '' then
        Kept := Kept + ', ';
      Kept := Kept + Name;
    end;
  end;
  if Kept <> '' then
    SaveStringToFile(AddBackslash(App) + KeptNote, 'NitroLearn was uninstalled, and your bots and settings were kept in this folder (' + Kept + ').' + #13#10#13#10 + 'To use them again, install NitroLearn into this same folder: NitroLearn-Setup accepts it.' + #13#10 + 'To get rid of them, delete this folder.' + #13#10, False);
end;

{ NitroLearn's own folders an uninstall could not delete (a file in them was in use), or ''. }
function LeftBehind(const App: String): String;
var
  Rest, Name: String;
  Comma: Integer;
begin
  Result := '';
  Rest := '.venv,.tools,python,tools,crates,';
  while Rest <> '' do
  begin
    Comma := Pos(',', Rest);
    Name := Copy(Rest, 1, Comma - 1);
    Rest := Copy(Rest, Comma + 1, Length(Rest));
    if DirExists(AddBackslash(App) + Name) then
    begin
      if Result <> '' then
        Result := Result + ', ';
      Result := Result + Name;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Rest, Name, App, Left: String;
  Comma: Integer;
begin
  App := ExpandConstant('{app}');
  if CurUninstallStep = usUninstall then
    KeepUserFiles := SuppressibleMsgBox('Keep your bots and settings?' + #13#10#13#10 + 'Yes: your trained bots, settings files, packed bots and replays (the folders runs, my-settings, packs, replays and bots) stay in ' + App + '.' + #13#10 + 'No: they are deleted with NitroLearn.' + #13#10#13#10 + 'RLBot is a separate program and is not touched.', mbConfirmation, MB_YESNO, IDYES) <> IDNO;
  if CurUninstallStep = usPostUninstall then
  begin
    if KeepUserFiles then
      WriteKeptNote(App)
    else
    begin
      Rest := UserFolders + ',';
      while Rest <> '' do
      begin
        Comma := Pos(',', Rest);
        Name := Copy(Rest, 1, Comma - 1);
        Rest := Copy(Rest, Comma + 1, Length(Rest));
        DelTree(AddBackslash(App) + Name, True, True, True);
      end;
    end;
    RemoveDir(App);
    Left := LeftBehind(App);
    if Left <> '' then
      SuppressibleMsgBox('Some of NitroLearn''s files could not be deleted (' + Left + '): a program was still using them.' + #13#10#13#10 + 'Restart the PC, then delete those folders in ' + App + ' yourself.', mbError, MB_OK, IDOK);
  end;
  if (CurUninstallStep = usDone) and KeepUserFiles then
    SuppressibleMsgBox('Your bots and settings are still in ' + App + '.' + #13#10#13#10 + 'Install NitroLearn into that folder again and they are there, as you left them.', mbInformation, MB_OK, IDOK);
end;
